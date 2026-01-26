#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-20:00      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=24G          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_edep_v2_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_edep_v2_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 		# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

## Train Conditional LDM without Latents
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/protons64-ldm-kl_no_latent.yaml \
# 	--logdir gaussian_blobs_pos_emb \
# 	--scale_lr False \
# 	--log_wandb True \
# 	--train \
# 	--gpus 0,

## Train Conditional LDM 
conda run -n ldm python3 -u main.py \
	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
	--logdir edep_protons64_v2_ldm \
	--scale_lr False \
	--log_wandb True \
	--train \
	--gpus 0,

	# --resume protons64sqrt_3cond/runs/checkpoints/last.ckpt \



## Get Inputs, Latents, Recos (~30 min for full val dataset)
# conda run -n ldm python3 -u main.py \
# python3 main.py \
# 	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
# 	--resume protons64x_div_3cond/runs/checkpoints/last.ckpt \
# 	--logdir zzz_p64v2_latents \
# 	--scale_lr False \
# 	--save_latents zzz_sample/train \
# 	--plot_latents False \
# 	--log_wandb False \
# 	--train \
# 	--gpus 0,

## Unconditional Sample LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,
	# --custom_latents protons64_ae_posteriors_gen_unscaled.npy\


## Flags 
# "conda run -n ldm" == run within conda enviroment ldm 
# "python3 -u" = run without buffering output 

## Submission partition (SBATCH -p)
# gpu_requeue
# gpu_test
# gpu 
# iaifi_gpu

## GPU Request 2 A100
# SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2

## Memory needed (for 64x64x1)
# 7000 (works for training)
# 6000 (mem crash for latents?) 