#!/bin/bash
#SBATCH --job-name=arc-ddp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=long
#SBATCH --time=24:00:00

# Export master address and port for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Add current directory to PYTHONPATH so python can find the 'src' module
export PYTHONPATH=$PWD

# Run the DDP training script using torchrun
# We use --nproc_per_node=8 to match the 8 GPUs requested
uv run torchrun --nproc_per_node=8 src/train_ddp.py