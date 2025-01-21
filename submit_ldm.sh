#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-05:30      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=8000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_cond_class_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_cond_class_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 		# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

## -m trace --listfuncs
## -m trace --trace
## -m pdb
# python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir zzz_protons64_testing \
# 	--scale_lr False \
# 	--log_wandb False \
# 	--train \
# 	--gpus 0,

## Train Conditional LDM 
# conda run -n ldm python3 -u main.py \
python3 -u main.py \
	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
	--logdir zxy_cond_ldm \
	--scale_lr False \
	--log_wandb False \
	--train \
	--gpus 0,


	# --resume protons64_cond_ldm_exp_t/2024-12-09T19-28-30_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \

	# --resume protons64_cond_ldm_v0/2024-11-05T00-36-04_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
	# --resume protons64_cond_ldm_test/2024-10-29T14-32-08_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \


## Train LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_ldm_l2_logs \
# 	--scale_lr False \
# 	--train \
# 	--gpus 0,1

## Get Inputs, Latents, Recos (~30 min for val on gpu_test)
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
# 	--resume protons64_ldm_logs/2024-09-24T22-43-29_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir zzz_log2 \
# 	--scale_lr False \
# 	--save_latents "zzz_sample2" \
# 	--plot_latents True \
# 	--log_wandb False \
# 	--train \
# 	--gpus 0,

## Sample LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,

	# --custom_latents protons64_ae_posteriors_gen_unscaled.npy\


	# --save_latents protons64_ae_disc_latents_test \
	# --save_latents zz_test \
	# --plot_latents False \
# protons64_ae_disc_post_test
	# --custom_latents protons64_ae_disc_latents_gen_unscaled.npy\

	# --resume ztest_lartpc64_ldm_logs/2024-07-30T20-04-59_lartpc_64-ldm-kl-8/checkpoints/N-Step-Checkpoint_epoch=0_global_step=2000.ckpt \
	# --logdir protons64_ldm_logs \


## Sample LDM 
# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume /n/home11/zimani/latent-diffusion/lartpc64_ldm_logs_v3/2024-06-12T01-47-26_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir lartpc64_sample_logs \
# 	--n_samples 800 \
# 	--batch_size 16 \



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