#!/bin/bash
#SBATCH -c 2               		# Number of cores (-c)
#SBATCH -t 0-06:00         		# Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu					# Partition to submit to
#SBATCH --mem=6000       		# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zae_%j.out  			# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zae_%j.err  			# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:2			# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@60		# Terminate program @x seconds before time limit 

conda run -n ldm python3 -u main.py \
	--base configs/autoencoder/autoencoder_kl_lartpc_64.yaml \
	--logdir protons64_ae_logs \
	--train \
	--gpus 0,1

## --resume lartpc64_logs_v2/checkpoints/last.ckpt \
