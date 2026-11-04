# LArTPC Latent Diffusion 

Documentation is still a work in progress.

## Original 
Paper: [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)<br/>
GitHub: [**Latent Diffusion**](https://github.com/CompVis/latent-diffusion)

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

## Running the model

There are three parts to running the model: autoencoder, diffusion, and sampling. I use shell scripts to run each of these either on an interactive node using `./submit_ldm.sh` or submit as a job `sbatch submit_ldm.sh`. 

Each dataset has its own configs and submit scripts:

| Dataset | Autoencoder config / script | LDM config / script |
| --- | --- | --- |
| protons64 | `configs/autoencoder/autoencoder_kl_protons64_16x16x3.yaml` / `submit_ae.sh` | `configs/latent-diffusion/protons64-ldm-kl.yaml` / `submit_ldm.sh` |
| electrons64 | `configs/autoencoder/autoencoder_kl_electrons64_16x16x3.yaml` / `submit_ae_electrons.sh` | `configs/latent-diffusion/electrons64-ldm-kl.yaml` / `submit_ldm_electrons.sh` |

### 1. Autoencoder 
Modify the autoencoder config for your dataset (see table above).

Train the autoencoder using its submit script.

Use Weights and Biases logging to determine the best checkpoint.  

Logs and checkpoints are saved to `<logdir>/<START_DATE_AND_TIME>_<config_name>/`, which contains `checkpoints/` (including `last.ckpt`) and `configs/` (the merged config used for the run). To resume, pass `--resume` either the run directory (loads `checkpoints/last.ckpt`) or a specific `.ckpt` file inside `<run>/checkpoints/`. The saved run configs are loaded automatically. `--resume` cannot be combined with `--name`.

### 2. Diffusion Model
Modify the LDM config for your dataset (see table above) to specify the desired autoencoder checkpoint (`ckpt_path` under `first_stage_config`).  

Train the latent diffusion model using its submit script.

### Trained checkpoints
The trained checkpoints are kept on lab storage under `/n/holystore01/LABS/iaifi_lab/Users/zimani/ldm_ckpts/`, not in the run directories:

| Dataset | Model | Checkpoint (relative to `ldm_ckpts/`) |
| --- | --- | --- |
| protons64 (edep) | Autoencoder (16x16x3) | `protons64_ae16x16x3/epoch=000120.ckpt` |
| protons64 (edep) | LDM | `protons64_cLDM/epoch=000075.ckpt` |
| electrons64 | Autoencoder (16x16x3) | `electrons64_ae16x16x3/checkpoints/epoch=000145.ckpt` |
| electrons64 | LDM | `electrons64_cLDM/checkpoints/epoch=000150.ckpt` |

Each folder also has a `configs/` folder holding the run's saved training config. Each LDM was trained on the autoencoder above it, and the LDM configs point to those autoencoders. For protons, the inference loop, sampling and renoising scripts all use the epoch 75 LDM. The electron checkpoint evaluation scripts are in `testing_electrons64/`.

### 3. Generate events 
Generate events using `submit_sample.sh`, which runs `run_condLDM.py`. Specify the momentum prompt using the flags px, py, pz. 

Each of the `--n_iters` batches is saved as `<outdir>/batch_<n>.npy`, with `--n_samples` events per batch. Set `--H 64 --W 64` for protons64 (the script defaults to 256).

Sampling quality, speed and diversity are controlled by:
- `--scale`: classifier-free guidance scale. Higher values follow the momentum condition more closely but give less diverse samples.
- `--ddim_steps`: more steps generally give better samples. Returns diminish above ~250.
- `--ddim_eta`: `0.0` gives deterministic DDIM, which keeps good quality at low step counts.
- `--plms`: use the PLMS sampler instead of DDIM, which allows even fewer steps.

## Adding a dataset 
Create a new dataloader class. The proton and electron loaders (`electrons64Train` / `electrons64Validation`) are all in `ldm/data/protons64.py`. Update the autoencoder and diffusion config files to point to the data files. 

## Analysis Scripts 
This will be in a seperate repo (coming soon). 


## Script Flags  

main.py flags: 
	- base: path to config yaml file 
	- logdir: directory to log checkpoints and outputs (default `logs`), also used as the Weights and Biases project name
	- log_wandb: boolean to use Weights and Biases (default True)
	- train: include to train the model 
	- gpus: GPU indices to use, must have a trailing comma (e.g. `--gpus 0,`)
	- resume: run directory or checkpoint to resume from (see above)
	- scale_lr: scale the base learning rate by ngpu * batch_size * n_accumulate (default True, LDM training uses False)
	- save_latents / plot_latents: save (and optionally plot) inputs, latents and reconstructions instead of training
	- seed: random seed (default 23)

Running the models 

1. Training the Autoencoder 
	- Create new config in configs/autoencoder
	- For protons64 dataset use autoencoder_kl_protons64_16x16x3.yaml
		- Uses KL divergence to (64,64,1) -> (16,16,3) 
	- Run the model use submit.ae to the grid 

## Acknowledgements
- This repo is a fork of [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).
- The diffusion code there builds on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
- The transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers).

## BibTeX
```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

