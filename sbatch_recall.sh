#!/bin/bash
#SBATCH --job-name=33_Contrecall
#SBATCH --partition=gpu-small-lp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=recall6.9breltr.log
#SBATCH --error=recall6.9breltr.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G


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

    echo "Running task with model_size=$model_size, truncated=$truncated, cuda_id=$cuda_id, relative=$relative dataset_idx=$dataset_idx batch_size=$batch_size"

    python runing_both.py \
        --relative $relative \
        --model_size $model_size \
        --truncated $truncated \
        --dataset_name local_all \
        --min_len 0 \
        --cuda $cuda_id \
        --dataset_idx $dataset_idx \
        --batch_size $batch_size \
        --gray recall
}

# 启动并行任务
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel relative 6.9b truncated 0 0 1" &


# 等待所有任务结束
wait