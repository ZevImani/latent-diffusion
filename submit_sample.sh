#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-00:10      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test					# Partition to submit to
#SBATCH --mem=6000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_sample0_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_sample0_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 	# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 


export CUDA_VISIBLE_DEVICES=0

## Unconditional Sampling (need modify config)
# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs/2024-09-24T22-43-29_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir protons64_sample \
# 	--n_samples 50000 \
# 	--batch_size 256 \
# 	--custom_steps 200

## Conditional Sampling 
# conda run -n ldm python3 scripts/txt2img.py \
# python3 scripts/txt2img.py \
# 	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
# 	--resume protons64_cond_class/runs/checkpoints/last.ckpt \
# 	--outdir zzz_sample \
# 	--px "389.8" \
# 	--py "-245.2" \
# 	--pz "-32.53" \
# 	--ddim_steps 50 \
# 	--ddim_eta 0.5 \
# 	--W 64 \
# 	--H 64 \
# 	--n_samples 16 \
# 	--n_iter 1 \
# 	--scale 5.0
	
	## Down right diag " 360.0, 168.6, -66.06"
	## Down left diag: " 389.8, -245.2, -32.53"
	# --resume protons64_cond_ldm_x/2024-11-18T16-04-15_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \


	# --resume protons64_cond_ldm_x/2024-11-18T16-04-15_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \


	## Normalized Momentum 
	# --resume protons64_cond_ldm_v1/2024-11-12T23-06-03_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
	
	## Bimodal Momentum (Px)
	# --resume protons64_cond_ldm_x/2024-11-13T11-24-13_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \


## Pretrained
python3 scripts/txt2img_orig.py 


## Sample LDM - using main.py
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train 


# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs_v3/2024-08-30T01-25-12_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir protons64_ldm_sample_logs \
# 	--vanilla_sample True \
# 	--n_samples 50000 \
# 	--batch_size 128 

	#  -c <\#ddim steps> -e <\#eta> 

## Sample LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,


## Train LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_ldm_logs_v3 \
# 	--scale_lr False \
# 	--train \
# 	--gpus 0,1

	# --save_latents protons64_ae_disc_latents_test \
	# --save_latents zz_test \
	# --plot_latents False \
# protons64_ae_disc_post_test
	# --custom_latents protons64_ae_disc_latents_gen_unscaled.npy\

	# --resume ztest_lartpc64_ldm_logs/2024-07-30T20-04-59_lartpc_64-ldm-kl-8/checkpoints/N-Step-Checkpoint_epoch=0_global_step=2000.ckpt \
	# --logdir protons64_ldm_logs \


## Sample LDM 
# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs_v3/2024-08-30T01-25-12_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir lartpc64_sample_logs \
# 	--n_samples 800 \
# 	--batch_size 16 \


# python3 -u scripts/sample_diffusion.py --resume protons64_ldm_logs_v3/2024-08-30T01-25-12_lartpc_64-ldm-kl-8/checkpoints/last.ckpt --logdir lartpc64_sample_logs --n_samples 800 --batch_size 16 


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