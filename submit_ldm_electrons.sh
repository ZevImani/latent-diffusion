#!/bin/bash
#SBATCH -c 3               		# Number of cores (-c): main process + 2 dataloader workers (in-memory data, mostly idle)
#SBATCH -t 0-14:00         		# Runtime in D-HH:MM (estimate ~4.2 min/epoch -> ~10.5 h for 150 epochs on a full A100; last.ckpt saved every epoch)
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=12G       		# Memory pool for all cores: measured peak ~9 GiB (4.3 GiB in-memory dataset shared by workers + model/CUDA + workers + wandb)
#SBATCH -o zldm_electrons64_%j.out  			# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_electrons64_%j.err  			# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1			# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120		# Terminate program @x seconds before time limit

## TF32 for matmuls (cross-attention/linear layers): +37% img/s. Convs already use TF32 by default.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

## Train Conditional LDM (momentum -> electron image) on the trained electron AE
conda run -n ldm python3 -u main.py \
	--base configs/latent-diffusion/electrons64-ldm-kl.yaml \
	--logdir electrons64_ldm \
	--scale_lr False \
	--log_wandb True \
	--train \
	--gpus 0,

## Resume
# conda run -n ldm python3 -u main.py \
# 	--resume electrons64_ldm/<run>/checkpoints/last.ckpt \
# 	--logdir electrons64_ldm \
# 	--scale_lr False \
# 	--log_wandb True \
# 	--train \
# 	--gpus 0,
