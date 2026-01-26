import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import matplotlib.pyplot as plt 

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def load_model_from_config(config, ckpt, verbose=False):
	"""Load model from config and checkpoint."""
	# print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model


def generate_conditioned_samples(
	config_path,
	checkpoint_path,
	px=389.8,
	py=-245.2,
	pz=-32.53,
	outdir="outputs/txt2img-samples",
	ddim_steps=200,
	use_plms=False,
	ddim_eta=0.0,
	n_iters=1,
	height=256,
	width=256,
	n_samples=4,
	scale=5.0,
	blobs=False,
	save_plot=True,
	plot_filename="cond_samples/cond_samples.png",
	verbose=False
):
	"""
	Generate conditioned samples using a diffusion model.
	
	Args:
		config_path (str): Path to the config YAML file
		checkpoint_path (str): Path to the model checkpoint
		px, py, pz (float): Momentum condition values
		outdir (str): Output directory for results
		ddim_steps (int): Number of DDIM sampling steps
		use_plms (bool): Whether to use PLMS sampling instead of DDIM
		ddim_eta (float): DDIM eta parameter
		n_iters (int): Number of batches to generate
		height, width (int): Image dimensions
		n_samples (int): Number of samples per batch
		scale (float): Unconditional guidance scale
		save_plot (bool): Whether to save a plot of the results
		plot_filename (str): Filename for the saved plot
		verbose (bool): Whether to print verbose model loading info
		
	Returns:
		numpy.ndarray: Generated samples array
	"""
	
	# Load config and model
	config = OmegaConf.load(config_path) 
	model = load_model_from_config(config, checkpoint_path, verbose=verbose)

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	# Initialize sampler
	if use_plms:
		sampler = PLMSSampler(model)
	else:
		sampler = DDIMSampler(model)

	# Create output directory
	os.makedirs(outdir, exist_ok=True)

	# Construct and normalize momentum vector 
	if isinstance(px, list):
		assert len(px) == len(pz) == len(pz)
		n_samples = len(px)
		prompts = torch.tensor(list(zip(px, py, pz)), dtype=torch.float32).to(device)
		prompts = prompts / 500 
	else: 
		momentum = [float(px), float(py), float(pz)]
		prompt = torch.tensor(momentum, dtype=torch.float32).to(device)
		prompt = prompt / 500 
		prompts = prompt.repeat(n_samples, 1)


	disable_tqdm = True 
	if verbose:
		print(f"Prompts shape: {prompts.shape}")
		print(f"Using embedding model: {model.cond_stage_model}")
		print(f"Using prompt: {prompt}")
		disable_tqdm = False

	all_samples = []

	with torch.no_grad():
		with model.ema_scope():
			for n in trange(n_iters, desc="Sampling", disable=disable_tqdm):

				# Tokenize and embed condition prompt 
				c = model.cond_stage_model(prompts)

				# Null prompt (unconditional condition)
				uc = "" 

				shape = [model.model.diffusion_model.in_channels,
						model.model.diffusion_model.image_size,
						model.model.diffusion_model.image_size]

				samples_ddim, _ = sampler.sample(
					S=ddim_steps,
					conditioning=c,
					batch_size=n_samples,
					shape=shape,
					verbose=False,
					unconditional_guidance_scale=scale,
					unconditional_conditioning=uc,
					eta=ddim_eta
				)

				x_samples_ddim = model.decode_first_stage(samples_ddim)
				x_samples_ddim = x_samples_ddim.cpu().squeeze()

				all_samples.append(x_samples_ddim.numpy())

				# Save individual batches if multiple iterations
				if n_iters > 1:
					np.save(os.path.join(outdir, f"batch_{n}.npy"), x_samples_ddim.numpy())

	# Combine all samples
	if len(all_samples) == 1:
		final_samples = all_samples[0]
	else:
		final_samples = np.concatenate(all_samples, axis=0)

	# Plot grid of generated samples if requested
	if save_plot and n_iters == 1:
		os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
		
		grid_size = 4
		fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
		axes = axes.ravel()

		# Scale plotting by max value in batch
		max_val = final_samples.max()
		if verbose:
			print(f"Max value: {max_val}")
		
		if isinstance(px, list): 
			fig.suptitle(f"Multiple Prompts", fontsize=20) 
		else: 
			fig.suptitle(f"Cond: {momentum}", fontsize=20)
		for i in range(min(grid_size**2, len(final_samples))):
			axes[i].imshow(final_samples[i], cmap='gray', interpolation='none', vmax=max_val)
			axes[i].axis('off')

		plt.tight_layout()
		plt.savefig(plot_filename)
		if verbose:
			print(f"Saved: {plot_filename}")
		plt.close()

	return final_samples


def main():
	"""Main function that preserves the original script's argument parsing."""
	import argparse
	
	parser = argparse.ArgumentParser()

	parser.add_argument("--px", type=str, default="389.8", help="P_x momentum condition")
	parser.add_argument("--py", type=str, default="-245.2", help="P_y momentum condition")
	parser.add_argument("--pz", type=str, default="-32.53", help="P_z momentum condition")
	parser.add_argument("--outdir", type=str, default="outputs/txt2img-samples", help="dir to write results to")
	parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
	parser.add_argument("--plms", action='store_true', help="use plms sampling")
	parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta")
	parser.add_argument("--n_iters", type=int, default=1, help="how many batches of samples")
	parser.add_argument("--H", type=int, default=256, help="image height, in pixel space")
	parser.add_argument("--W", type=int, default=256, help="image width, in pixel space")
	parser.add_argument("--n_samples", type=int, default=128, help="how many samples to produce")
	parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
	parser.add_argument("-r", "--resume", type=str, required=True, help="path to checkpoint")
	parser.add_argument("-b", "--base", type=str, required=True, help="config yaml file")

	opt = parser.parse_args()

	# Call the function with parsed arguments
	samples = generate_conditioned_samples(
		config_path=opt.base,
		checkpoint_path=opt.resume,
		px=float(opt.px),
		py=float(opt.py),
		pz=float(opt.pz),
		outdir=opt.outdir,
		ddim_steps=opt.ddim_steps,
		use_plms=opt.plms,
		ddim_eta=opt.ddim_eta,
		n_iters=opt.n_iters,
		height=opt.H,
		width=opt.W,
		n_samples=opt.n_samples,
		scale=opt.scale,
		verbose=True
	)
	
	print(f"Generated {len(samples)} samples with shape {samples[0].shape if len(samples) > 0 else 'N/A'}")
	print(f"Saved to: {opt.outdir}")


if __name__ == "__main__":

	main() 

	# x = [314.0, 292.5, -382.6, -391.4]
	# y = [-126.4, 130.7, 2.8, -99.7]
	# z = [249.1, 70.8, -16.3, -217.9]
	# x = 314.0
	# y = -126.4
	# z = 249.1
	# batch = generate_conditioned_samples(
	# 	px=x,py=y,pz=z,
	# 	n_samples=2,
	# 	n_iters=1, 
	# 	config_path="/n/home11/zimani/latent-diffusion/configs/latent-diffusion/protons64-ldm-kl.yaml",
	# 	checkpoint_path="/n/home11/zimani/latent-diffusion/edep_protons64_ldm/runs/checkpoints/epoch=000040.ckpt",
	# 	save_plot=True,
	# 	verbose=False)
	# print(batch.shape)