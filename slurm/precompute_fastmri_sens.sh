#!/bin/bash
#SBATCH --job-name=fmri_sens
#SBATCH --gres=gpu:1
#SBATCH --partition=radiology
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=outputs/sens_%A_%a.out
#SBATCH --error=outputs/sens_%A_%a.err

set -euo pipefail

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

cd /gpfs/data/chopralab/zhangs18/fmri

export PYTHONPATH=code

R="${R:?Set R=4 or R=8}"

python code/scripts/precompute_fastmri_sens_cache.py \
    --acceleration "$R" \
    --num_volumes 30 \
    --seed 20260904 \
    --num_shards 8 \
    --shard_id "$SLURM_ARRAY_TASK_ID"
