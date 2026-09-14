#!/bin/bash
#SBATCH --job-name=sham_cache_r4_va
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=radiology
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham_cache_r4_va_%A_%a.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham_cache_r4_va_%A_%a.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

S=${SLURM_ARRAY_TASK_ID}

PYTHONPATH=code \
python code/scripts/cache_shamaei_fastmri256_espirit.py \
    --acceleration 4 \
    --split validation \
    --num_shards 8 \
    --shard_id ${S} \
    --resume
