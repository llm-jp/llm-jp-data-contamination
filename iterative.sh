#!/bin/bash

# List of model sizes to iterate over
model_sizes=("160m" "410m" "1b" "2.8b" "6.9b" "12b")

# Function to submit a job with a specific model size
submit_job() {
  local model_size=$1

  # Create a temporary script with the current model size
  temp_script="temp_${model_size}.sh"

  # Copy the original script content with updated model size
  cat <<EOF > "$temp_script"
#!/bin/bash
#SBATCH --job-name=33_Contrecall
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=recall${model_size}relatr.log
#SBATCH --error=recall${model_size}relatr.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G

# 创建日志目录
mkdir -p logs

# 出错时退出脚本
set -e

# 定义并行化函数
srun_parallel () {
  local relative=\$1
  local model_size=\$2
  local truncated=\$3
  local cuda_id=\$4
  local dataset_idx=\$5
  local batch_size=\$6
  echo "Running task with model_size=\$model_size, truncated=\$truncated, cuda_id=\$cuda_id, relative=\$relative dataset_idx=\$dataset_idx batch_size=\$batch_size"
  python runing_both.py \\
  --relative \$relative \\
  --model_size \$model_size \\
  --truncated \$truncated \\
  --dataset_name local_all \\
  --min_len 0 \\
  --cuda \$cuda_id \\
  --dataset_idx \$dataset_idx \\
  --batch_size \$batch_size \\
  --gray recall
}

# 启动并行任务
srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -c "\$(declare -f srun_parallel); srun_parallel relative ${model_size} truncated 0 0 2" &

# 等待所有任务结束
wait
EOF

  # Submit the job
  sbatch "$temp_script"

  # Remove the temporary script
  rm "$temp_script"
}

# Iterate over each model size and submit a job
for size in "${model_sizes[@]}"; do
  submit_job "$size"
done