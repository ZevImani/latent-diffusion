#!/bin/bash
#SBATCH -c 2                	# Number of cores (-c)
#SBATCH -t 0-08:30      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu					# Partition to submit to
#SBATCH --mem=7000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:2		 	# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

## Train LDM 
conda run -n ldm python3 -u main.py \
	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
	--logdir protons64_ldm_logs \
	--train \
	--gpus 0,1

	# --resume ztest_lartpc64_ldm_logs/2024-07-30T20-04-59_lartpc_64-ldm-kl-8/checkpoints/N-Step-Checkpoint_epoch=0_global_step=2000.ckpt \


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

## GPU Request 2 A100
# SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2

## Memory needed (for 64x64x1)
# 7000 (works for training)
# 6000 (mem crash for latents?) 