#!/bin/bash
#SBATCH --account=oyx@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=tp
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

module purge
source $WORK/litenv/bin/activate


# python3 -m scripts.tokenize_parallel --dataset the_stack --limit 1% --key content
python3 -m scripts.tokenize_parallel --dataset c4 --limit 10% --key text
