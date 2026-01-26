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

### 1. Autoencoder 
Modify the config file `configs/autoencoder/autoencoder_kl_protons64_16x16x3.yaml`.

Train the autoencoder using `submit_ae.sh`

Use Weights and Biases logging to determine the best checkpoint.  

### 2. Diffusion Model
Modify the config file: `configs/latent-diffusion/protons64-ldm-kl.yaml` to specify the desired autoencoder checkpoint.  

Train the latent diffusion model using `submit_ldm.sh`

### 3. Generate events 
Generate events using `submit_sample.sh`. Specify the momentum prompt using the flags px, py, pz. 

## Adding a dataset 
Create a new dataloader. Currently using `ldm\data\protons64.py`. Update the autoencoder and diffusion config files to point to the data files. 

## Analysis Scripts 
This will be in a seperate repo (coming soon). 


## Script Flags  

main.py flags: 
	- Base: path to config yaml file 
	- Logdir: directory to log checkpoints, outputcs, also used as weights and biases name
	- log_wandb: boolean to use weights and biases 
	- Train: include to train the model 
	- gpus: number of gpus to use, must have trailling comma 

Running the models 

1. Training the Autoencoder 
	- Create new config in configs/autoencoder
	- For protons64 dataset use autoencoder_kl_protons64_16x16x3.yaml
		- Uses KL divergence to (64,64,1) -> (16,16,3) 
	- Run the model use submit.ae to the grid 

