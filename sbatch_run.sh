#!/bin/bash
#SBATCH --job-name=DataContamination12b
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=72:00:00
#SBATCH --output=trunc12bj.log
#SBATCH --error=trunc12bj.err
#SBATCH --cpus-per-task=8


# Run your script
srun python runing_both.py \
  --relative absolute \
  --model_size 12b \
  --truncated truncated \
  --dataset_name local_all \
  --min_len 0 \
  --cuda 0 \
  --refer_cuda 1 \
  --dataset_idx 1 \
  --batch_size 1