#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:32gb:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6


uv run python src/idea.py
