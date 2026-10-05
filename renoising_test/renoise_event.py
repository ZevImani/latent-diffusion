#!/usr/bin/env python3
"""
Single-event renoise round-trip test for the latent diffusion model.

Takes a single image from the validation set, encodes it, noises it to
t=T_RENOISE via the forward diffusion process, then denoises back to t=0
using DDIM. A working model should approximately recover the original.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config
from ldm.data.protons64 import edepProtons64Validation
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "../configs/latent-diffusion/protons64-ldm-kl.yaml")
DEFAULT_CKPT = "/n/home11/zimani/latent-diffusion/edep_protons64_v2_ldm/runs/checkpoints/last.ckpt"
DEFAULT_KINKED  = "/n/home11/zimani/inference_loop/datasets/kinked_tracks/kinked_tracks_top64.npz"
INFERENCE_DIR   = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../inference_loop


def spatial_emd(img_recon, img_orig, blur=0.01):
    """
    Sinkhorn Wasserstein-1 between two (H, W) images treated as 2D mass
    distributions over pixel positions.  Pixel intensities are clamped >= 0
    and normalized to sum to 1 before transport.  Returns a scalar float.
    """
    img_recon = img_recon.squeeze()
    img_orig  = img_orig.squeeze()

    orig_sum = img_orig.clamp(min=0).sum()
    if orig_sum == 0:
        return 0.0

    H, W = img_orig.shape
    ys = torch.arange(H, dtype=torch.float32, device=img_orig.device)
    xs = torch.arange(W, dtype=torch.float32, device=img_orig.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

    w_recon = img_recon.flatten().clamp(min=0)
    w_orig  = img_orig.flatten().clamp(min=0)
    w_recon = w_recon / (w_recon.sum() + 1e-8)
    w_orig  = w_orig  / (w_orig.sum()  + 1e-8)

    loss = SamplesLoss("sinkhorn", p=1, blur=blur)(w_recon, positions, w_orig, positions)
    return loss.item()


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.eval().to(device)
    return model


def ddim_denoise(model, z_noisy, t_renoise, cond, device, ddim_steps=100, eta=0.0,
                 n_marginal=0, marginal_chunk=1):
    """
    Denoise z_noisy (noised to t_renoise) back to t=0 using a custom sub-schedule
    covering [1, t_renoise].

    n_marginal=0 (default): standard DDIM using the provided 'cond'.
    n_marginal>0: at each step, average e_t over n_marginal random momenta
    sampled uniformly from [-1, 1] (normalized = [-500, 500] MeV), marginalizing
    out the conditioning.  'cond' is ignored in this mode.
    marginal_chunk: UNet batch size for each marginal forward pass (default 1).
    Larger values are faster but use more GPU memory.
    """
    n_steps = min(ddim_steps, t_renoise)
    if n_steps <= 0:
        return z_noisy

    ddim_timesteps = np.unique(np.round(np.linspace(1, t_renoise, n_steps)).astype(int))

    alphas_cumprod_np = model.alphas_cumprod.detach().cpu().numpy()
    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
        alphacums=alphas_cumprod_np,
        ddim_timesteps=ddim_timesteps,
        eta=eta,
        verbose=False,
    )

    to_torch = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
    ddim_sigmas_t         = to_torch(ddim_sigmas)
    ddim_alphas_t         = to_torch(ddim_alphas)
    ddim_alphas_prev_t    = to_torch(ddim_alphas_prev)
    ddim_sqrt_one_minus_t = to_torch(np.sqrt(1.0 - ddim_alphas))

    if n_marginal == 0:
        sampler = DDIMSampler(model)
        sampler.ddim_timesteps             = ddim_timesteps
        sampler.ddim_sigmas                = ddim_sigmas_t
        sampler.ddim_alphas                = ddim_alphas_t
        sampler.ddim_alphas_prev           = ddim_alphas_prev_t
        sampler.ddim_sqrt_one_minus_alphas = ddim_sqrt_one_minus_t
        print(f"  DDIM schedule: {len(ddim_timesteps)} steps, "
              f"t={ddim_timesteps[-1]} → {ddim_timesteps[0]}")
        z_denoised, _ = sampler.ddim_sampling(cond, shape=z_noisy.shape, x_T=z_noisy)
        return z_denoised

    # Marginalisation path: average e_t over n_marginal random momenta per step
    print(f"  Marginal DDIM: {len(ddim_timesteps)} steps, K={n_marginal}, "
          f"t={ddim_timesteps[-1]} → {ddim_timesteps[0]}")
    img         = z_noisy.clone()
    b           = img.shape[0]
    total_steps = len(ddim_timesteps)

    for i, step in enumerate(np.flip(ddim_timesteps)):
        index = total_steps - i - 1
        ts    = torch.full((b,), int(step), device=device, dtype=torch.long)

        rand_moms  = torch.rand(n_marginal, 3, device=device) * 2.0 - 1.0
        rand_conds = model.get_learned_conditioning(rand_moms)  # (K, 3, 16)

        e_t = torch.zeros_like(img)
        for start in range(0, n_marginal, marginal_chunk):
            end   = min(start + marginal_chunk, n_marginal)
            chunk = end - start
            e_t  += model.apply_model(
                img.expand(chunk, -1, -1, -1),
                ts.expand(chunk),
                rand_conds[start:end],
            ).sum(dim=0, keepdim=True)
        e_t = e_t / n_marginal

        a_t        = torch.full((b, 1, 1, 1), ddim_alphas_t[index],         device=device)
        a_prev     = torch.full((b, 1, 1, 1), ddim_alphas_prev_t[index],    device=device)
        sigma      = torch.full((b, 1, 1, 1), ddim_sigmas_t[index],         device=device)
        sqrt_1m_at = torch.full((b, 1, 1, 1), ddim_sqrt_one_minus_t[index], device=device)

        pred_x0 = (img - sqrt_1m_at * e_t) / a_t.sqrt()
        dir_xt  = (1.0 - a_prev - sigma**2).sqrt() * e_t
        img     = a_prev.sqrt() * pred_x0 + dir_xt + sigma * torch.randn_like(img)

    return img


def main():
    parser = argparse.ArgumentParser(description="Single-event renoise round-trip test for LDM")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to model config YAML")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to model checkpoint")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index into the validation dataset")
    parser.add_argument("--t_renoise", type=float, default=0.5,
                        help="Noise level to apply: float in [0.0, 1.0] (0=clean, 1=pure noise)")
    parser.add_argument("--ddim_steps", type=int, default=100,
                        help="Number of DDIM steps for denoising")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta: 0=deterministic, 1=full stochastic")
    parser.add_argument("--blur", type=float, default=0.01,
                        help="Sinkhorn blur for EMD (default 0.01)")
    parser.add_argument("--out", default="plot_renoise_event.png",
                        help="Output plot filename (saved next to this script)")
    parser.add_argument("--kinked", action="store_true",
                        help="Load from kinked-tracks dataset instead of the validation set")
    parser.add_argument("--kinked_path", default=DEFAULT_KINKED,
                        help="Path to kinked_tracks.npz")
    parser.add_argument("--infer_momentum", action="store_true",
                        help="Run gradient inference on the true image and use the inferred "
                             "momentum as the LDM conditioning instead of the dataset label")
    parser.add_argument("--infer_n_iter", type=int, default=39,
                        help="Number of SGD iterations for momentum inference (default 39)")
    parser.add_argument("--n_marginal", type=int, default=0,
                        help="If > 0, average e_t over this many random momentum conditionings "
                             "at each DDIM step (marginalizes out the conditioning). "
                             "Default 0 = use the dataset/inferred momentum as-is.")
    parser.add_argument("--marginal_chunk", type=int, default=1,
                        help="UNet batch size per marginal forward pass (default 1 = minimal memory). "
                             "Increase to trade memory for speed when n_marginal > 0.")
    parser.add_argument("--latents", action="store_true",
                        help="Add latent-channel rows below each decoded panel in the output PNG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading model from:\n  {args.ckpt}")
    model = load_model(args.config, args.ckpt, device)
    print(f"  scale_factor = {model.scale_factor:.4f}")
    print(f"  num_timesteps = {model.num_timesteps}")
    t_renoise = min(int(round(args.t_renoise * (model.num_timesteps - 1))), model.num_timesteps - 1)

    # --- Load one sample ---
    if args.kinked:
        print(f"\nLoading kinked-tracks sample {args.sample_idx} from:\n  {args.kinked_path}")
        kt = np.load(args.kinked_path)
        image_np    = kt["images"][args.sample_idx][:, :, np.newaxis]  # (64, 64) → (64, 64, 1)
        momentum_np = kt["momenta"][args.sample_idx] / 500.0           # MeV → normalized
    else:
        print(f"\nLoading validation sample {args.sample_idx} ...")
        dataset = edepProtons64Validation()
        sample = dataset[args.sample_idx]
        image_np    = sample["image"]     # (64, 64, 1) float32
        momentum_np = sample["momentum"]  # (3,) float32, already divided by 500

    print(f"  Image shape: {image_np.shape}  range: [{image_np.min():.3f}, {image_np.max():.3f}]")

    # --- Optionally run gradient inference to get conditioning momentum ---
    true_p_mev = momentum_np * 500  # always the dataset label, kept for display
    if args.infer_momentum:
        sys.path.insert(0, INFERENCE_DIR)
        from run_inference import run_inference as _run_inference
        print(f"\nRunning gradient inference ({args.infer_n_iter} iterations) ...")
        infer_results = _run_inference(
            target_img=image_np.squeeze(),
            true_momentum=tuple(float(v) for v in true_p_mev),
            n_iterations=args.infer_n_iter,
            device=device,
            save_plots=False,
            verbose=True,
        )
        best_idx = int(np.argmin(infer_results["dist_path"]))
        best_mom = infer_results["mom_path"][best_idx]  # (px, py, pz) in MeV
        momentum_np = np.array(best_mom, dtype=np.float32) / 500.0
        print(f"\n  True momentum (MeV):     ({true_p_mev[0]:.1f}, {true_p_mev[1]:.1f}, {true_p_mev[2]:.1f})")
        print(f"  Inferred momentum (MeV): ({momentum_np[0]*500:.1f}, {momentum_np[1]*500:.1f}, {momentum_np[2]*500:.1f})")
    else:
        print(f"  Momentum (MeV): px={momentum_np[0]*500:.1f}  py={momentum_np[1]*500:.1f}  pz={momentum_np[2]*500:.1f}")

    # (64, 64, 1) → (1, 1, 64, 64)
    x0 = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    momentum = torch.tensor(momentum_np, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        with model.ema_scope():

            # --- Encode to latent ---
            z0 = model.get_first_stage_encoding(model.encode_first_stage(x0))
            print(f"\n  Latent z0 shape: {z0.shape}")

            # --- Forward diffusion: noise to t_renoise ---
            t_tensor = torch.full((1,), t_renoise, device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            z_noisy = model.q_sample(x_start=z0, t=t_tensor, noise=noise)

            # Fraction of signal remaining at t_renoise
            snr_t = model.sqrt_alphas_cumprod[t_renoise].item()
            print(f"  Signal fraction at t={t_renoise}: sqrt(alpha_bar) = {snr_t:.4f}")

            # --- Embed momentum (cross-attention conditioning) ---
            cond = model.get_learned_conditioning(momentum)  # (1, 3, 16)

            # --- Reverse: DDIM denoise from t_renoise → 0 ---
            print(f"\nDenoising: t={t_renoise} → 0  ({args.ddim_steps} DDIM steps, eta={args.ddim_eta})")
            z_denoised = ddim_denoise(
                model, z_noisy, t_renoise, cond, device,
                ddim_steps=args.ddim_steps, eta=args.ddim_eta,
                n_marginal=args.n_marginal,
                marginal_chunk=args.marginal_chunk,
            )

            # --- Decode all three latents to pixel space ---
            img_orig  = model.decode_first_stage(z0).cpu().squeeze().numpy()
            img_noisy = model.decode_first_stage(z_noisy).cpu().squeeze().numpy()
            img_recon = model.decode_first_stage(z_denoised).cpu().squeeze().numpy()

            if args.latents:
                z0_lat         = z0.cpu().numpy()[0]           # (C, H', W')
                z_noisy_lat    = z_noisy.cpu().numpy()[0]
                z_denoised_lat = z_denoised.cpu().numpy()[0]

    # --- Metrics ---
    T    = model.num_timesteps - 1  # 999
    diff = img_orig - img_recon
    mae  = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())

    orig_t  = torch.tensor(img_orig,  dtype=torch.float32).to(device)
    recon_t = torch.tensor(img_recon, dtype=torch.float32).to(device)
    emd = spatial_emd(recon_t, orig_t, blur=args.blur)

    print(f"\nReconstruction metrics (pixel space):")
    print(f"  MAE:  {mae:.5f}")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  EMD:  {emd:.5f}")
    print(f"  True image range:    [{img_orig.min():.4f}, {img_orig.max():.4f}]")
    print(f"  Reconstructed range: [{img_recon.min():.4f}, {img_recon.max():.4f}]")

    # --- Plot ---
    t_norm       = t_renoise / T
    vmax         = max(img_orig.max(), img_recon.max(), 1e-9)
    abs_diff_max = max(np.abs(diff).max(), 1e-9)

    if args.latents:
        n_chan = z0_lat.shape[0]
        fig, all_axes = plt.subplots(
            1 + n_chan, 4,
            figsize=(16, 4 + 2.2 * n_chan),
            gridspec_kw={"height_ratios": [2.0] + [1.0] * n_chan},
            squeeze=False,
        )
        axes = all_axes[0]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_orig, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
    axes[0].set_title("True image")
    axes[0].axis("off")

    axes[1].imshow(img_noisy, cmap="gray", interpolation="none")
    axes[1].set_title(f"Noisy  T = {t_norm:.2f}")
    axes[1].axis("off")

    axes[2].imshow(img_recon, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
    axes[2].set_title(f"Reconstructed\nEMD = {emd:.4f}")
    axes[2].axis("off")

    im = axes[3].imshow(diff, cmap="bwr", interpolation="none",
                        vmin=-abs_diff_max, vmax=abs_diff_max)
    axes[3].set_title(f"Residual (orig - recon)\nMAE={mae:.4f}  RMSE={rmse:.4f}")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if args.latents:
        lat_triplet = [z0_lat, z_noisy_lat, z_denoised_lat]
        lat_titles  = ["z_orig", f"z_noisy (T={t_norm:.2f})", "z_denoised"]
        for c in range(n_chan):
            vabs = max(*(float(np.abs(lat[c]).max()) for lat in lat_triplet), 1e-9)
            for col, (lat, lbl) in enumerate(zip(lat_triplet, lat_titles)):
                ax = all_axes[c + 1, col]
                ax.imshow(lat[c], cmap="RdBu_r", vmin=-vabs, vmax=vabs, interpolation="none")
                ax.axis("off")
                if c == 0:
                    ax.set_title(lbl, fontsize=8)
            all_axes[c + 1, 0].set_ylabel(f"ch {c}", fontsize=8, rotation=0, labelpad=28, va="center")
            all_axes[c + 1, 3].axis("off")

    p_cond = momentum_np * 500
    if args.n_marginal > 0:
        mom_line = f"marginal K={args.n_marginal} (conditioning averaged out)"
    elif args.infer_momentum:
        mom_line = (f"p_true=({true_p_mev[0]:.1f},{true_p_mev[1]:.1f},{true_p_mev[2]:.1f})  "
                    f"p_cond=({p_cond[0]:.1f},{p_cond[1]:.1f},{p_cond[2]:.1f}) MeV [inferred]")
    else:
        mom_line = f"p = ({p_cond[0]:.1f}, {p_cond[1]:.1f}, {p_cond[2]:.1f}) MeV"
    fig.suptitle(
        f"Renoise round-trip  |  t ={t_norm:.2f}  |  "
        f"{args.ddim_steps} DDIM steps  |  eta={args.ddim_eta}  |  blur={args.blur}\n"
        f"{mom_line}   [signal fraction {snr_t:.3f}]",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, args.out)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
