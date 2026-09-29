#!/usr/bin/env python3
"""
EMD vs noise level sweep for the LDM renoise round-trip test.

For each noise level t in [0, t_max] at intervals of t_step:
  1. Encode a validation image to the latent space.
  2. Add noise via q_sample to timestep t.
  3. Denoise back to t=0 with DDIM.
  4. Compute Sinkhorn Wasserstein-1 (spatial EMD) between original and reconstructed.

Saves all reconstructed images + EMD values to a .npz archive and plots
EMD vs noise level.

Use --plot_only to regenerate plots from a saved .npz without rerunning inference.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LDM_DIR     = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LDM_DIR)

from ldm.util import instantiate_from_config
from ldm.data.protons64 import edepProtons64Validation
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters


DEFAULT_CONFIG  = os.path.join(LDM_DIR, "configs/latent-diffusion/protons64-ldm-kl.yaml")
DEFAULT_CKPT    = "/n/home11/zimani/latent-diffusion/edep_protons64_v2_ldm/runs/checkpoints/last.ckpt"
DEFAULT_KINKED  = "/n/home11/zimani/inference_loop/datasets/line_kinked_tracks/line_kinked_tracks_64.npz"
INFERENCE_DIR   = os.path.dirname(LDM_DIR)  # .../inference_loop


def norm_to_timestep(val, num_timesteps):
    """Convert a normalized [0, 1] noise level to an integer DDPM timestep."""
    return min(int(round(val * (num_timesteps - 1))), num_timesteps - 1)


# ---------------------------------------------------------------------------
# Model / inference helpers
# ---------------------------------------------------------------------------

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    pl_sd  = torch.load(ckpt_path, map_location="cpu")
    model  = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.eval().to(device)
    return model


def ddim_denoise(model, z_noisy, t_renoise, cond, device, ddim_steps=100, eta=0.0,
                 n_marginal=0, marginal_chunk=1):
    """
    Denoise z_noisy (noised to t_renoise) back to t=0 using DDIMSampler with a
    uniform sub-schedule over [1, t_renoise].  Returns z_noisy unchanged if
    t_renoise == 0 (nothing to denoise).

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

    alphas_np = model.alphas_cumprod.detach().cpu().numpy()
    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
        alphacums=alphas_np,
        ddim_timesteps=ddim_timesteps,
        eta=eta,
        verbose=False,
    )

    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    ddim_sigmas_t         = to_t(ddim_sigmas)
    ddim_alphas_t         = to_t(ddim_alphas)
    ddim_alphas_prev_t    = to_t(ddim_alphas_prev)
    ddim_sqrt_one_minus_t = to_t(np.sqrt(1.0 - ddim_alphas))

    if n_marginal == 0:
        sampler = DDIMSampler(model)
        sampler.ddim_timesteps             = ddim_timesteps
        sampler.ddim_sigmas                = ddim_sigmas_t
        sampler.ddim_alphas                = ddim_alphas_t
        sampler.ddim_alphas_prev           = ddim_alphas_prev_t
        sampler.ddim_sqrt_one_minus_alphas = ddim_sqrt_one_minus_t
        z_denoised, _ = sampler.ddim_sampling(cond, shape=z_noisy.shape, x_T=z_noisy)
        return z_denoised

    # Marginalisation path: average e_t over n_marginal random momenta per step
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
    positions = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # (H*W, 2)

    w_recon = img_recon.flatten().clamp(min=0)
    w_orig  = img_orig.flatten().clamp(min=0)
    w_recon = w_recon / (w_recon.sum() + 1e-8)
    w_orig  = w_orig  / (w_orig.sum()  + 1e-8)

    loss = SamplesLoss("sinkhorn", p=1, blur=blur)(w_recon, positions, w_orig, positions)
    return loss.item()


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

def run_iterate(model, z0, img_orig, momentum_np_init, p_true_mev,
                n_iterations, t_renoise, device,
                ddim_steps, ddim_eta, n_marginal, marginal_chunk,
                blur, infer_n_iter):
    """
    Iterative refinement loop.

    Each iteration:
      1. Noise z0 to t_renoise.
      2. Denoise using current conditioning momentum.
      3. Decode → compute EMD vs original.
      4. Run gradient inference on the denoised image → update conditioning momentum.

    Returns a dict with arrays indexed 0..n_iterations where index 0 is the
    initial state (no denoise yet):
      momenta_mev  (N+1, 3)    momentum conditioning at the start of each iteration
      emds         (N+1,)      EMD of each reconstruction vs original (0 at index 0)
      images       (N+1, H, W) reconstructed images (original at index 0)
    """
    sys.path.insert(0, INFERENCE_DIR)
    from run_inference import run_inference as _run_inference

    img_orig_np    = img_orig.cpu().numpy()
    current_mom_np = momentum_np_init.copy()  # normalized (/ 500)

    momenta_mev = [current_mom_np * 500]
    emds        = [0.0]
    images      = [img_orig_np]
    latents     = [z0.cpu().numpy()]

    for i in range(n_iterations):
        print(f"\n--- Iterate {i + 1}/{n_iterations} ---")
        print(f"  Conditioning: px={current_mom_np[0]*500:.1f}  "
              f"py={current_mom_np[1]*500:.1f}  pz={current_mom_np[2]*500:.1f} MeV")

        # ---- noise → denoise ----
        momentum_t = torch.tensor(current_mom_np, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            with model.ema_scope():
                cond     = model.get_learned_conditioning(momentum_t)
                t_tensor = torch.full((1,), t_renoise, device=device, dtype=torch.long)
                z_noisy  = model.q_sample(x_start=z0, t=t_tensor, noise=torch.randn_like(z0))
                z_den    = ddim_denoise(model, z_noisy, t_renoise, cond, device,
                                        ddim_steps=ddim_steps, eta=ddim_eta,
                                        n_marginal=n_marginal, marginal_chunk=marginal_chunk)
                img_recon = model.decode_first_stage(z_den).squeeze()
                latents.append(z_den.cpu().numpy())
                del z_noisy, z_den
                torch.cuda.empty_cache()

        img_recon_np = img_recon.cpu().numpy()
        emd          = spatial_emd(img_recon, img_orig, blur=blur)
        print(f"  EMD: {emd:.5f}")

        images.append(img_recon_np)
        emds.append(emd)

        # ---- gradient inference on denoised image ----
        print(f"  Running inference ({infer_n_iter} iterations) ...")
        true_mom_tuple = (tuple(float(v) for v in p_true_mev)
                          if p_true_mev is not None else (0.0, 0.0, 0.0))
        infer_results  = _run_inference(
            target_img    = img_recon_np,
            true_momentum = true_mom_tuple,
            n_iterations  = infer_n_iter,
            device        = device,
            save_plots    = False,
            verbose       = False,
        )
        best_idx       = int(np.argmin(infer_results["dist_path"]))
        best_mom       = infer_results["mom_path"][best_idx]   # MeV
        current_mom_np = np.array(best_mom, dtype=np.float32) / 500.0
        momenta_mev.append(current_mom_np * 500)

        print(f"  Updated momentum: px={current_mom_np[0]*500:.1f}  "
              f"py={current_mom_np[1]*500:.1f}  pz={current_mom_np[2]*500:.1f} MeV")

    return {
        "momenta_mev" : np.array(momenta_mev),
        "emds"        : np.array(emds),
        "images"      : np.array(images),
        "latents"     : np.array(latents),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(timestep_array, emd_values,
               img_orig_np, reconstructed_imgs,
               p, sample_idx, ddim_steps, blur,
               img_step, num_timesteps,
               out_plot, out_grid,
               p_true=None, n_marginal=0,
               latents=None):
    """
    Generate and save both plots from pre-computed sweep data.

    Parameters
    ----------
    timestep_array     : (T,) int   — raw DDPM timesteps tested
    emd_values         : (T,) float — EMD at each timestep
    img_orig_np        : (H, W)     — original decoded image
    reconstructed_imgs : (T, H, W)  — denoised images
    p                  : (3,)       — momentum in MeV
    sample_idx         : int
    ddim_steps         : int
    blur               : float
    img_step           : int        — raw-timestep spacing for image grid columns
    num_timesteps      : int        — total DDPM timesteps (1000)
    out_plot           : str        — path for EMD plot
    out_grid           : str        — path for image grid plot
    """
    T      = num_timesteps - 1          # 999
    t_norm = timestep_array / T         # [0, 1]
    t_max  = int(timestep_array[-1])
    t_step = int(timestep_array[1] - timestep_array[0]) if len(timestep_array) > 1 else t_max

    midpoint = len(timestep_array) // 2
    print(f"\nEMD at t=0.00: {emd_values[0]:.5f}")
    print(f"EMD at t={timestep_array[midpoint]/T:.2f}: {emd_values[midpoint]:.5f}")
    print(f"EMD at t=1.00: {emd_values[-1]:.5f}")

    # --- EMD vs noise level ---
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t_norm, emd_values, "o-", markersize=3, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Noise level T")
    ax.set_ylabel("EMD")
    ax.set_title("EMD vs Noise Level")
    ax.grid(True, alpha=0.3)

    if n_marginal > 0:
        mom_line = f"marginal K={n_marginal} (conditioning averaged out)"
    elif p_true is not None:
        mom_line = (f"p_true=({p_true[0]:.1f},{p_true[1]:.1f},{p_true[2]:.1f})  "
                    f"p_cond=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) MeV [inferred]")
    else:
        mom_line = f"p = ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) MeV"
    fig.suptitle(
        f"Renoise sweep  |  sample={sample_idx}  |  "
        f"t_step={t_step / T:.3f}  |  {ddim_steps} DDIM steps  |  blur={blur}\n"
        f"{mom_line}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_plot}")

    # --- Image grid ---
    display_ts = list(np.arange(img_step, t_max + 1, img_step, dtype=int))
    if t_max not in display_ts:
        display_ts.append(t_max)

    display_indices = [int(np.argmin(np.abs(timestep_array - t))) for t in display_ts]
    display_ts      = [int(timestep_array[i]) for i in display_indices]

    n_cols = 1 + len(display_indices)
    vmax   = max(img_orig_np.max(), reconstructed_imgs[display_indices].max(), 1e-9)

    if latents is not None:
        _lat_list = [latents[0]] + [latents[i] for i in display_indices]
        lat_all = np.stack([l[0] if l.ndim == 4 else l for l in _lat_list])  # (n_cols, C, H', W')
        n_chan_lat = lat_all.shape[1]
        fig2, all_ax2 = plt.subplots(
            1 + n_chan_lat, n_cols,
            figsize=(2.2 * n_cols, 3.0 + 2.0 * n_chan_lat),
            gridspec_kw={"height_ratios": [2.0] + [1.0] * n_chan_lat},
            squeeze=False,
        )
        axes2 = all_ax2[0]
    else:
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3.0))
        if n_cols == 1:
            axes2 = [axes2]

    axes2[0].imshow(img_orig_np, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
    axes2[0].set_title("True image", fontsize=9)
    axes2[0].axis("off")

    for col, (arr_idx, t_disp) in enumerate(zip(display_indices, display_ts)):
        ax = axes2[col + 1]
        ax.imshow(reconstructed_imgs[arr_idx], cmap="gray", vmin=0, vmax=vmax,
                  interpolation="none")
        ax.set_title(f"t = {t_disp / T:.2f}\nEMD = {emd_values[arr_idx]:.3f}", fontsize=8)
        ax.axis("off")

    if latents is not None:
        for c in range(n_chan_lat):
            vabs = max(float(np.abs(lat_all[:, c]).max()), 1e-9)
            for col in range(n_cols):
                ax = all_ax2[c + 1, col]
                ax.imshow(lat_all[col, c], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                          interpolation="none")
                ax.axis("off")
            all_ax2[c + 1, 0].set_ylabel(f"ch {c}", fontsize=8, rotation=0,
                                           labelpad=28, va="center")

    cond_label = (f"marginal K={n_marginal}" if n_marginal > 0
                  else f"p = ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) MeV")
    fig2.suptitle(
        f"True image vs denoised (every {img_step / T:.2f} noise)  |  "
        f"sample={sample_idx}\n"
        f"{cond_label}  |  {ddim_steps} DDIM steps",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_grid}")


def make_iterate_plots(iter_results, p_true_mev, sample_idx, t_renoise,
                       num_timesteps, ddim_steps, blur, out_emd, out_grid,
                       latents=None):
    """
    Generate plots for iterative refinement mode.

    out_emd  : EMD + momentum convergence plot path
    out_grid : image grid plot path
    """
    momenta = iter_results["momenta_mev"]   # (N+1, 3)
    emds    = iter_results["emds"]          # (N+1,)
    images  = iter_results["images"]        # (N+1, H, W)
    n_iter  = len(emds) - 1
    iters   = np.arange(n_iter + 1)

    t_norm = t_renoise / (num_timesteps - 1)

    # --- EMD + momentum convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax0 = axes[0]
    ax0.plot(iters[1:], emds[1:], "o-", markersize=5, color="steelblue")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("EMD")
    ax0.set_title(f"EMD vs Iteration  (t={t_norm:.2f}, {ddim_steps} DDIM steps, blur={blur})")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    labels = ["px", "py", "pz"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for j, (lbl, col) in enumerate(zip(labels, colors)):
        ax1.plot(iters, momenta[:, j], "o-", markersize=4, label=lbl, color=col)
    if p_true_mev is not None:
        for j, (lbl, col) in enumerate(zip(labels, colors)):
            ax1.axhline(p_true_mev[j], linestyle="--", color=col, alpha=0.5,
                        label=f"{lbl} true")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Momentum (MeV)")
    ax1.set_title("Momentum Conditioning vs Iteration")
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)

    true_label = (f"p_true=({p_true_mev[0]:.1f},{p_true_mev[1]:.1f},{p_true_mev[2]:.1f}) MeV"
                  if p_true_mev is not None else "")
    fig.suptitle(
        f"Iterative refinement  |  sample={sample_idx}  |  t={t_norm:.2f}\n{true_label}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_emd, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_emd}")

    # --- Image grid ---
    labels_grid = ["Original"] + [f"Iter {k}" for k in range(1, n_iter + 1)]
    n_cols      = len(images)
    vmax        = max(images.max(), 1e-9)

    if latents is not None:
        lat_sq = latents[:, 0] if latents.ndim == 5 else latents  # (N+1, C, H', W')
        n_chan_lat = lat_sq.shape[1]
        fig2, all_ax2 = plt.subplots(
            1 + n_chan_lat, n_cols,
            figsize=(2.2 * n_cols, 3.0 + 2.0 * n_chan_lat),
            gridspec_kw={"height_ratios": [2.0] + [1.0] * n_chan_lat},
            squeeze=False,
        )
        axes2 = all_ax2[0]
    else:
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3.0))
        if n_cols == 1:
            axes2 = [axes2]

    for col, (img, lbl, emd) in enumerate(zip(images, labels_grid, emds)):
        axes2[col].imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
        title = lbl if col == 0 else f"{lbl}\nEMD={emd:.3f}"
        axes2[col].set_title(title, fontsize=8)
        axes2[col].axis("off")

    if latents is not None:
        for c in range(n_chan_lat):
            vabs = max(float(np.abs(lat_sq[:, c]).max()), 1e-9)
            for col in range(n_cols):
                ax = all_ax2[c + 1, col]
                ax.imshow(lat_sq[col, c], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                          interpolation="none")
                ax.axis("off")
            all_ax2[c + 1, 0].set_ylabel(f"ch {c}", fontsize=8, rotation=0,
                                           labelpad=28, va="center")

    fig2.suptitle(
        f"Iterative refinement  |  sample={sample_idx}  |  t={t_norm:.2f}\n"
        f"p_init=({momenta[0,0]:.1f},{momenta[0,1]:.1f},{momenta[0,2]:.1f}) MeV",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_grid}")


def make_sweep_latent_grid(timestep_array, latents, img_step, num_timesteps, out_path):
    """
    Plot each latent channel for selected sweep timesteps.
    Rows = latent channels, columns = z_orig then z_denoised at img_step intervals.
    No EMD — latent values can be negative.
    """
    T     = num_timesteps - 1
    t_max = int(timestep_array[-1])

    display_ts = list(np.arange(img_step, t_max + 1, img_step, dtype=int))
    if t_max not in display_ts:
        display_ts.append(t_max)
    display_indices = [int(np.argmin(np.abs(timestep_array - t))) for t in display_ts]
    display_ts      = [int(timestep_array[i]) for i in display_indices]

    # latents[0] is z0 (t=0); latents[i] is z_denoised at timestep_array[i]
    selected = np.stack([latents[i] for i in display_indices])   # (n_sel, C, H', W')
    all_lat  = np.concatenate([latents[:1], selected], axis=0)   # (n_sel+1, C, H', W')
    labels   = ["z_orig"] + [f"t={t/T:.2f}" for t in display_ts]

    n_cols = len(all_lat)
    n_chan = all_lat.shape[1]

    fig, axes = plt.subplots(n_chan, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_chan),
                             squeeze=False)

    for c in range(n_chan):
        chan_data = all_lat[:, c, :, :]
        vabs      = float(np.abs(chan_data).max()) or 1.0
        for col in range(n_cols):
            ax = axes[c][col]
            ax.imshow(chan_data[col], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                      interpolation="none")
            ax.axis("off")
            if c == 0:
                ax.set_title(labels[col], fontsize=8)
        axes[c][0].set_ylabel(f"ch {c}", fontsize=8, rotation=0, labelpad=28, va="center")

    fig.suptitle(
        f"Latents per channel  |  sweep  |  every {img_step / T:.2f} noise",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def make_iterate_latent_grid(iter_results, sample_idx, t_renoise, num_timesteps, out_latent_grid):
    """
    Plot each latent channel for every iteration.
    Rows = latent channels, columns = original z0 then z_denoised per iteration.
    No EMD — latent values can be negative.
    """
    latents    = iter_results["latents"]   # (N+1, C, H', W')
    n_iter     = latents.shape[0] - 1
    n_chan     = latents.shape[1]
    t_norm     = t_renoise / (num_timesteps - 1)
    n_cols     = n_iter + 1
    col_labels = ["z_orig"] + [f"z iter {k}" for k in range(1, n_iter + 1)]

    fig, axes = plt.subplots(n_chan, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_chan),
                             squeeze=False)

    for c in range(n_chan):
        chan_data = latents[:, c, :, :]          # (N+1, H', W')
        vabs      = float(np.abs(chan_data).max()) or 1.0
        for col in range(n_cols):
            ax = axes[c][col]
            ax.imshow(chan_data[col], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                      interpolation="none")
            ax.axis("off")
            if c == 0:
                ax.set_title(col_labels[col], fontsize=8)
        axes[c][0].set_ylabel(f"ch {c}", fontsize=8, rotation=0, labelpad=28, va="center")

    fig.suptitle(
        f"Latents per channel  |  sample={sample_idx}  |  t={t_norm:.2f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_latent_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_latent_grid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Renoise sweep: EMD vs noise level for LDM")
    parser.add_argument("--config",     default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt",       default=DEFAULT_CKPT)
    parser.add_argument("--sample_idx", type=int,   default=0,
                        help="Validation dataset sample index")
    parser.add_argument("--t_step",     type=float, default=0.01,
                        help="Noise-level increment between sweep points as a fraction of [0,1] "
                             "(default 0.01 ≈ 10 DDPM steps)")
    parser.add_argument("--t_max",      type=float, default=1.0,
                        help="Maximum noise level as a fraction of [0,1] (default 1.0 = full noise)")
    parser.add_argument("--ddim_steps", type=int,   default=100,
                        help="DDIM denoising steps per noise level")
    parser.add_argument("--ddim_eta",   type=float, default=0.0,
                        help="DDIM eta: 0=deterministic, 1=full stochastic")
    parser.add_argument("--blur",       type=float, default=0.01,
                        help="Sinkhorn blur for EMD (default 0.01)")
    parser.add_argument("--img_step",   type=float, default=0.1,
                        help="Show denoised image every img_step noise fraction in grid "
                             "(default 0.1 ≈ every 100 DDPM steps)")
    parser.add_argument("--out_npy",    default="renoise_sweep.npz",
                        help="Output / input numpy archive (saved next to this script)")
    parser.add_argument("--out_plot",   default="plot_emd_vs_noise.png",
                        help="Output EMD plot filename")
    parser.add_argument("--out_grid",   default="plot_image_grid.png",
                        help="Output image grid plot filename")
    parser.add_argument("--plot_only",  action="store_true",
                        help="Load existing .npz and regenerate plots without running inference")
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
    parser.add_argument("--iterate",    type=int,   default=0,
                        help="Number of iterative refinement cycles: noise the event to "
                             "--iterate_t (or --t_max), denoise, run inference to update the "
                             "momentum conditioning, and repeat. Default 0 = standard sweep mode.")
    parser.add_argument("--iterate_t",  type=float, default=None,
                        help="Noise level as a fraction of [0,1] for iterate mode "
                             "(e.g. 0.5 = timestep 500). Defaults to --t_max when not specified.")
    parser.add_argument("--latents", action="store_true",
                        help="Add latent-channel rows below each decoded panel in the image grid PNG")
    args = parser.parse_args()

    out_stem  = os.path.splitext(args.out_npy)[0]
    out_dir   = os.path.join(SCRIPT_DIR, out_stem)
    os.makedirs(out_dir, exist_ok=True)

    npy_path  = os.path.join(out_dir, args.out_npy)
    plot_path = os.path.join(out_dir, args.out_plot)
    grid_path = os.path.join(out_dir, args.out_grid)

    # ------------------------------------------------------------------
    # Plot-only path: load archive and skip all inference
    # ------------------------------------------------------------------
    if args.plot_only:
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"--plot_only requires an existing archive: {npy_path}")
        print(f"Loading archive: {npy_path}")
        data    = np.load(npy_path)
        _num_ts = int(data["num_timesteps"]) if "num_timesteps" in data else 1000

        if "iterate_t" in data:
            print("Detected iterate archive.")
            _iter_results = {"momenta_mev": data["momenta_mev"],
                             "emds":        data["emds"],
                             "images":      data["images"],
                             "latents":     data["latents"]}
            make_iterate_plots(
                iter_results  = _iter_results,
                p_true_mev    = data["p_dataset_mev"] if "p_dataset_mev" in data else None,
                sample_idx    = int(data["sample_idx"]),
                t_renoise        = int(data["iterate_t"]),
                num_timesteps = _num_ts,
                ddim_steps    = int(data["ddim_steps"]),
                blur          = float(data["blur"]),
                out_emd       = plot_path,
                out_grid      = grid_path,
                latents       = _iter_results["latents"] if (args.latents and "latents" in _iter_results) else None,
            )
            make_iterate_latent_grid(
                iter_results    = _iter_results,
                sample_idx      = int(data["sample_idx"]),
                t_renoise          = int(data["iterate_t"]),
                num_timesteps   = _num_ts,
                out_latent_grid = grid_path.replace(".png", "_latent.png"),
            )
        else:
            print("Detected sweep archive.")
            _img_step = max(1, norm_to_timestep(args.img_step, _num_ts))
            make_plots(
                timestep_array    = data["timesteps"],
                emd_values        = data["emds"],
                img_orig_np       = data["original"],
                reconstructed_imgs= data["reconstructed"],
                p                 = data["momentum_mev"],
                sample_idx        = int(data["sample_idx"]),
                ddim_steps        = int(data["ddim_steps"]),
                blur              = float(data["blur"]),
                img_step          = _img_step,
                num_timesteps     = _num_ts,
                out_plot          = plot_path,
                out_grid          = grid_path,
                p_true            = data["momentum_true_mev"] if "momentum_true_mev" in data else None,
                n_marginal        = int(data["n_marginal"]) if "n_marginal" in data else 0,
                latents           = data["latents"] if (args.latents and "latents" in data) else None,
            )
            if "latents" in data:
                make_sweep_latent_grid(
                    timestep_array = data["timesteps"],
                    latents        = data["latents"],
                    img_step       = _img_step,
                    num_timesteps  = _num_ts,
                    out_path       = grid_path.replace(".png", "_latent.png"),
                )
        return

    # ------------------------------------------------------------------
    # Full inference path
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model ...")
    model = load_model(args.config, args.ckpt, device)
    print(f"  scale_factor   = {model.scale_factor:.4f}")
    print(f"  num_timesteps  = {model.num_timesteps}")

    if args.kinked:
        print(f"\nLoading kinked-tracks sample {args.sample_idx} from:\n  {args.kinked_path}")
        kt = np.load(args.kinked_path)
        image_np    = kt["images"][args.sample_idx][:, :, np.newaxis]  # (64, 64) → (64, 64, 1)
        momentum_np = kt["momenta"][args.sample_idx] / 500.0           # MeV → normalized
    else:
        print(f"\nLoading validation sample {args.sample_idx} ...")
        dataset     = edepProtons64Validation()
        sample      = dataset[args.sample_idx]
        image_np    = sample["image"]     # (64, 64, 1) float32
        momentum_np = sample["momentum"]  # (3,) float32, divided by 500

    print(f"  Image range:    [{image_np.min():.3f}, {image_np.max():.3f}]")

    # --- Optionally run gradient inference to get conditioning momentum ---
    p_dataset_mev = momentum_np * 500  # raw dataset label, always preserved
    p_true = p_dataset_mev             # always the dataset label, kept for display
    if args.infer_momentum:
        sys.path.insert(0, INFERENCE_DIR)
        from run_inference import run_inference as _run_inference
        print(f"\nRunning gradient inference ({args.infer_n_iter} iterations) ...")
        infer_results = _run_inference(
            target_img=image_np.squeeze(),
            true_momentum=tuple(float(v) for v in p_true),
            n_iterations=args.infer_n_iter,
            device=device,
            save_plots=False,
            verbose=True,
        )
        best_idx = int(np.argmin(infer_results["dist_path"]))
        best_mom = infer_results["mom_path"][best_idx]  # (px, py, pz) in MeV
        momentum_np = np.array(best_mom, dtype=np.float32) / 500.0
        print(f"\n  True momentum (MeV):     ({p_true[0]:.1f}, {p_true[1]:.1f}, {p_true[2]:.1f})")
        print(f"  Inferred momentum (MeV): ({momentum_np[0]*500:.1f}, {momentum_np[1]*500:.1f}, {momentum_np[2]*500:.1f})")
    else:
        p_true = None  # signals make_plots that no inference was done
        print(f"  Momentum (MeV): px={momentum_np[0]*500:.1f}  py={momentum_np[1]*500:.1f}  pz={momentum_np[2]*500:.1f}")
    p = momentum_np * 500

    x0       = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    momentum = torch.tensor(momentum_np, dtype=torch.float32).unsqueeze(0).to(device)

    T        = model.num_timesteps
    t_max    = norm_to_timestep(args.t_max,    T)
    t_step   = max(1, norm_to_timestep(args.t_step,   T))
    img_step = max(1, norm_to_timestep(args.img_step, T))

    # ------------------------------------------------------------------
    # Iterate mode: noise → denoise → infer → repeat N times
    # ------------------------------------------------------------------
    if args.iterate > 0:
        iterate_t = (norm_to_timestep(args.iterate_t, T)
                     if args.iterate_t is not None else t_max)
        print(f"\nIterative refinement: {args.iterate} iterations at t={iterate_t} "
              f"(t_norm={iterate_t / (T - 1):.3f})")

        with torch.no_grad():
            with model.ema_scope():
                z0       = model.get_first_stage_encoding(model.encode_first_stage(x0))
                img_orig = model.decode_first_stage(z0).squeeze()

        iter_results = run_iterate(
            model            = model,
            z0               = z0,
            img_orig         = img_orig,
            momentum_np_init = momentum_np,
            p_true_mev       = p_dataset_mev,
            n_iterations     = args.iterate,
            t_renoise           = iterate_t,
            device           = device,
            ddim_steps       = args.ddim_steps,
            ddim_eta         = args.ddim_eta,
            n_marginal       = args.n_marginal,
            marginal_chunk   = args.marginal_chunk,
            blur             = args.blur,
            infer_n_iter     = args.infer_n_iter,
        )

        latent_grid_path = grid_path.replace(".png", "_latent.png")

        np.savez(npy_path,
                 momenta_mev   = iter_results["momenta_mev"],
                 emds          = iter_results["emds"],
                 images        = iter_results["images"],
                 latents       = iter_results["latents"],
                 iterate_t     = np.array(iterate_t),
                 n_iterations  = np.array(args.iterate),
                 sample_idx    = np.array(args.sample_idx),
                 ddim_steps    = np.array(args.ddim_steps),
                 blur          = np.array(args.blur),
                 num_timesteps = np.array(model.num_timesteps),
                 p_dataset_mev = p_dataset_mev)
        print(f"\nSaved: {npy_path}")

        make_iterate_plots(
            iter_results  = iter_results,
            p_true_mev    = p_dataset_mev,
            sample_idx    = args.sample_idx,
            t_renoise        = iterate_t,
            num_timesteps = model.num_timesteps,
            ddim_steps    = args.ddim_steps,
            blur          = args.blur,
            out_emd       = plot_path,
            out_grid      = grid_path,
            latents       = iter_results["latents"] if args.latents else None,
        )
        make_iterate_latent_grid(
            iter_results  = iter_results,
            sample_idx    = args.sample_idx,
            t_renoise        = iterate_t,
            num_timesteps = model.num_timesteps,
            out_latent_grid = latent_grid_path,
        )
        return

    timestep_array = np.arange(0, t_max + 1, t_step, dtype=int)

    print(f"\nSweeping {len(timestep_array)} levels: "
          f"t in [{timestep_array[0]}, {timestep_array[-1]}], step={t_step}")
    print(f"DDIM steps per level: {args.ddim_steps}  |  blur={args.blur}")

    emd_values         = []
    reconstructed_imgs = []
    latents            = []

    with torch.no_grad():
        with model.ema_scope():

            z0       = model.get_first_stage_encoding(model.encode_first_stage(x0))
            cond     = model.get_learned_conditioning(momentum)
            img_orig = model.decode_first_stage(z0).squeeze()

            for t_renoise in tqdm(timestep_array, desc="Renoise sweep"):
                t_renoise = int(t_renoise)

                if t_renoise == 0:
                    img_recon = img_orig.clone()
                    latents.append(z0.cpu().numpy())
                else:
                    t_tensor   = torch.full((1,), t_renoise, device=device, dtype=torch.long)
                    z_noisy    = model.q_sample(x_start=z0, t=t_tensor,
                                                noise=torch.randn_like(z0))
                    z_denoised = ddim_denoise(model, z_noisy, t_renoise, cond, device,
                                              ddim_steps=args.ddim_steps,
                                              eta=args.ddim_eta,
                                              n_marginal=args.n_marginal,
                                              marginal_chunk=args.marginal_chunk)
                    img_recon  = model.decode_first_stage(z_denoised).squeeze()
                    latents.append(z_denoised.cpu().numpy())
                    del z_noisy, z_denoised
                    torch.cuda.empty_cache()

                emd = spatial_emd(img_recon, img_orig, blur=args.blur)
                emd_values.append(emd)
                reconstructed_imgs.append(img_recon.cpu().numpy())

    emd_values         = np.array(emd_values)
    reconstructed_imgs = np.array(reconstructed_imgs)
    latents            = np.array(latents)
    img_orig_np        = img_orig.cpu().numpy()
    signal_fractions   = model.sqrt_alphas_cumprod[timestep_array].cpu().numpy()

    npz_kwargs = dict(
        original         = img_orig_np,
        reconstructed    = reconstructed_imgs,
        latents          = latents,
        timesteps        = timestep_array,
        emds             = emd_values,
        signal_fractions = signal_fractions,
        momentum_mev     = p,
        sample_idx       = args.sample_idx,
        blur             = args.blur,
        ddim_steps       = args.ddim_steps,
        num_timesteps    = model.num_timesteps,
        n_marginal       = args.n_marginal,
    )
    if p_true is not None:
        npz_kwargs["momentum_true_mev"] = p_true
    np.savez(npy_path, **npz_kwargs)
    print(f"\nSaved: {npy_path}")
    print(f"  Keys: original {img_orig_np.shape}, "
          f"reconstructed {reconstructed_imgs.shape}, "
          f"timesteps {timestep_array.shape}, emds {emd_values.shape}")

    make_plots(
        timestep_array    = timestep_array,
        emd_values        = emd_values,
        img_orig_np       = img_orig_np,
        reconstructed_imgs= reconstructed_imgs,
        p                 = p,
        sample_idx        = args.sample_idx,
        ddim_steps        = args.ddim_steps,
        blur              = args.blur,
        img_step          = img_step,
        num_timesteps     = model.num_timesteps,
        out_plot          = plot_path,
        out_grid          = grid_path,
        p_true            = p_true,
        n_marginal        = args.n_marginal,
        latents           = latents if args.latents else None,
    )
    make_sweep_latent_grid(
        timestep_array = timestep_array,
        latents        = latents,
        img_step       = img_step,
        num_timesteps  = model.num_timesteps,
        out_path       = grid_path.replace(".png", "_latent.png"),
    )


if __name__ == "__main__":
    main()
