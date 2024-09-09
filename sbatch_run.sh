#!/bin/bash
#SBATCH --job-name=DataContaminationall
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:8              # 请求两个GPU (total out of 2 nodes which creates 4 tasks)
#SBATCH --nodes=1                 # 只需要一个节点
#SBATCH --time=72:00:00
#SBATCH --output=runall
#SBATCH --error=runall
#SBATCH --ntasks=4                # 总共运行四个任务
#SBATCH --cpus-per-task=4         # 每个任务分配4个CPU

# 创建日志目录

# 出错时退出脚本
set -e

# 实际运行的命令
srun_parallel () {
    local model_size=$1
    local truncated=$2
    local cuda_id=$3
    local refer_cuda_id=$4

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
srun --ntasks=1 bash -c "srun_parallel 12b truncated 0 7" &
srun --ntasks=1 bash -c "srun_parallel 12b untruncated 1 6" &
srun --ntasks=1 bash -c "srun_parallel 6.9b truncated 2 5" &
srun --ntasks=1 bash -c "srun_parallel 6.9b untruncated 3 4" &

# 等待所有任务结束
wait