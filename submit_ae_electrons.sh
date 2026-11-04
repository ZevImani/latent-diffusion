#!/bin/bash
#SBATCH -c 4               		# Number of cores (-c)
#SBATCH -t 2-00:00         		# Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=16G       		# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zae_electrons64_16x16x3_%j.out  			# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zae_electrons64_16x16x3_%j.err  			# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1			# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120		# Terminate program @x seconds before time limit 

## Train Autoencoder 
conda run -n ldm python3 -u main.py \
	--base configs/autoencoder/autoencoder_kl_electrons64_16x16x3.yaml \
	--logdir electrons64 \
	--log_wandb True \
	--train \
	--gpus 0,

## Show latents from a specific checkpoint (possibly deprecated)
# conda run -n ldm python3 -u main.py \
# 	--base configs/autoencoder/autoencoder_kl_protons64_8x8x3.yaml \
# 	--resume protons64_ae_8x8x3/2024-11-26T01-44-35_autoencoder_kl_protons64_8x8x3/checkpoints/last.ckpt \
# 	--logdir zzz_protons64_ae_8x8x3 \
# 	--log_wandb False \
# 	--save_latents "zzz123" \
# 	--plot_latents True \
# 	--train \
# 	--gpus 0,



## Flags Info
# "conda run -n ldm" = run within conda enviroment ldm 
# "python3 -u" = run without buffering output 

## Submission partition (SBATCH -p)
# gpu_requeue
# gpu_test
# gpu 
# iaifi_gpu
