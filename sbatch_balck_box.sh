#!/bin/bash
#SBATCH --job-name=33_ContBl
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=72:00:00
#SBATCH --output=Black.log
#SBATCH --error=Black.err
#SBATCH --cpus-per-task=8

# 创建日志目录
mkdir -p logs

# 出错时退出脚本
set -e

# 定义并行化函数
srun_parallel () {
    local relative=$1
    local model_size=$2
    local truncated=$3
    local cuda_id=$4
    local dataset_idx=$5
    local batch_size=$6


    echo "Running task with model_size=$model_size, truncated=$truncated, cuda_id=$cuda_id, refer_cuda_id=$refer_cuda_id relative=$relative dataset_idx=$dataset_idx batch_size=$batch_size"

    python runing_both.py \
        --relative $relative \
        --model_size $model_size \
        --truncated $truncated \
        --dataset_name local_all \
        --min_len 0 \
        --cuda $cuda_id \
        --refer_cuda 0 \
        --dataset_idx $dataset_idx \
        --generation_batch_size $batch_size  \
        --gray black
}

# 启动并行任务
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 12b truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 6.9b truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 2.8b  truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 1b truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 410m truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel absolute 160m truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel relative 12b truncated 0 0 16" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel relative 6.9b truncated 0 0 16" &


# 等待所有任务结束
wait