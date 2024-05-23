#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

#SBATCH --nodelist=slurm0-a3-ghpc-20
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1800GB
#SBATCH --job-name=dpo
#SBATCH --output=%x_%j.log

source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate dpo311

LD_LIBRARY_PATH=$EXP_HOME/miniconda3/envs/dpo311/lib CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch --config_file accelerate_configs/zero2.yaml train.py \
  --model="geniacllm/dMoE_8B_pretrain_0520_iter134999" \
  --wandb-project="dmoe-dpo" \
  --wandb-name="dMoE_8B_pretrain_0520_iter134999" \
