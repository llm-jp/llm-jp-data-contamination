#!/bin/bash
#SBATCH --job-name=DataContamination6.9b
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=6.9bj.log
#SBATCH --error=6.9bj.err
#SBATCH --cpus-per-task=8


# Run your script
srun python runing_both.py \
  --relative absolute \
  --model_size 6.9b \
  --truncated untruncated \
  --dataset_name local_all \
  --min_len 0 \
  --cuda 0 \
  --refer_cuda 1 \
  --dataset_idx 1 \
  --batch_size 2