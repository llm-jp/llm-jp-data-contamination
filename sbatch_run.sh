#!/bin/bash
#SBATCH --job-name=DataContaminationall
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=72:00:00
#SBATCH --output=logs/runall_%j.log
#SBATCH --error=logs/runaller_%j.err
#SBATCH --cpus-per-task=8

# 创建日志目录
mkdir -p logs

# 出错时退出脚本
set -e

# 定义并行化函数
srun_parallel () {
    local model_size=$1
    local truncated=$2
    local cuda_id=$3
    local refer_cuda_id=$4

    echo "Running task with model_size=$model_size, truncated=$truncated, cuda_id=$cuda_id, refer_cuda_id=$refer_cuda_id"

    python runing_both.py \
        --relative absolute \
        --model_size $model_size \
        --truncated $truncated \
        --dataset_name local_all \
        --min_len 0 \
        --cuda $cuda_id \
        --refer_cuda $refer_cuda_id \
        --dataset_idx 1 \
        --batch_size 1
}

# 启动并行任务
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel 12b truncated 0 7" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel 12b untruncated 1 6" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel 6.9b truncated 2 5" &
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "$(declare -f srun_parallel); srun_parallel 6.9b untruncated 3 4" &

# 等待所有任务结束
wait