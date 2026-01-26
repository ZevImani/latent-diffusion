#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-01:00      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=16000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zsamp_ldm_sample3_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zsamp_ldm_sample3_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 	# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

export CUDA_VISIBLE_DEVICES=0

# checkpoint_path="/edep_protons64/epoch=000040.ckpt"
# checkpoint_path="edep_protons64_ldm/runs/checkpoints/epoch=000040.ckpt"

checkpoint_path="edep_protons64_v2_ldm/runs/checkpoints/epoch=000075.ckpt"

	# --outdir one_mom_sample/ldm_epoch40_sample1 \

## Conditional Sampling 
# python3 scripts/txt2img.py \
conda run -n ldm python3 gen_cLDM.py \
	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
	--outdir ldm_protons_sample4/ \
	--px "-391.4" \
	--py "-99.7" \
	--pz "-217.9" \
	--ddim_steps 200 \
	--ddim_eta 0.0 \
	--W 64 \
	--H 64 \
	--n_samples 128 \
	--n_iters 100 \
	--scale 5.0 \
	--resume "$checkpoint_path"	

### 
# # Momentums from bimodal validation dataset 
# bimodal sample1 = (314.0, -126.4, 249.1)
# bimodal sample2 = (292.5, 130.7, 70.8)
# bimodal sample3 = (-382.6, 2.8, -16.3)
# bimodal sample4 = (-391.4, -99.7, -217.9)

# # Full Sample
# sampleA = 18.1, 242.8, 549.8
# sampleB = 19.6, -491.9, 533.7

# # Bad Momentum Reco 
# bad_reco = 199.1, 307.1, 382.2

# # Directions
# Down right diag " 360.0, 168.6, -66.06"
# Down left diag: " 389.8, -245.2, -32.53"
# Straight up: "-382.5, 2.785, -16.33
### 

## Unconditional Sampling (possibly deprecated)
# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs/2024-09-24T22-43-29_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir protons64_sample \
# 	--n_samples 50000 \
# 	--batch_size 256 \
# 	--custom_steps 200


## Sample LDM (possibly deprecated)
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,
