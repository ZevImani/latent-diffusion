#!/bin/bash
#SBATCH -c 2               		# Number of cores (-c)
#SBATCH -t 0-08:30         		# Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=10G       		# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zae_small_%j.out  			# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zae_small_%j.err  			# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1			# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120		# Terminate program @x seconds before time limit 

conda run -n ldm python3 -u main.py \
	--base configs/autoencoder/autoencoder_kl_lartpc_64.yaml \
	--logdir protons64_ae_disc_small_logs \
	--train \
	--gpus 0,

# --logdir protons64_ae_disc_logs \

## --resume lartpc64_logs_v2/checkpoints/last.ckpt \
	# --resume protons64_ae_disc_logs/2024-08-09T16-34-42_autoencoder_kl_lartpc_64 \



## Flags 
# "conda run -n ldm" = run within conda enviroment ldm 
# "python3 -u" = run without buffering output 

## Submission partition (SBATCH -p)
# gpu_requeue
# gpu_test
# gpu 
# iaifi_gpu

## Memory needed (for 64x64x1)
# 7000 (works for training)
# 6000 (mem crash for latents?) 