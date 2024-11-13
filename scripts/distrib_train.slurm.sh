#!/bin/bash
#SBATCH --nodes=1            # total number of nodes (N to be defined)
#SBATCH --ntasks-per-node=2  # number of tasks per node (here 4 tasks, or 1 task per GPU)
#SBATCH --gres=gpu:2         # number of GPUs reserved per node (here 4, or all the GPUs)
#SBATCH --cpus-per-task=10   # number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --time=01:00:00
#SBATCH --job-name=tg
#SBATCH --output=logs/%A_%x.out
#SBATCH --error=logs/%A_%x.err

mkdir -p logs
module purge
source $MY_ENV/bin/activate

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 2 \
  gan/train.py \
  --epochs 100 \
  --workers 8 \
  --batch_size 512
