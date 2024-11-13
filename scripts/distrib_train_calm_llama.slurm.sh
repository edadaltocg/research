#!/bin/bash
#SBATCH --account=oyx@a100
#SBATCH --nodes=1            # total number of nodes (N to be defined)
#SBATCH --ntasks-per-node=8  # number of tasks per node (here 4 tasks, or 1 task per GPU)
#SBATCH --gres=gpu:8         # number of GPUs reserved per node (here 4, or all the GPUs)
#SBATCH --cpus-per-task=8   # number of cores per task (4x10 = 40 cores, or all the core
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100
#SBATCH --job-name=tcl
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --array=0-2

module purge
source $WORK/litenv/bin/activate

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 8 \
  tutorials/trainer_class_2.py calm  \
  --experiment_idx $SLURM_ARRAY_TASK_ID