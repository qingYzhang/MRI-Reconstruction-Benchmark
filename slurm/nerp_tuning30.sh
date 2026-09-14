#!/bin/bash
#SBATCH --job-name=nerp1000
#SBATCH --gres=gpu:1
#SBATCH --partition=radiology
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=outputs/nerp_%A_%a.out
#SBATCH --error=outputs/nerp_%A_%a.err

set -euo pipefail

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

cd /gpfs/data/chopralab/zhangs18/fmri

export PYTHONPATH=code

R="${R:?Set R=4 or R=8}"

python code/scripts/eval_fastmri_prior_free.py \
    --subset tuning30 \
    --acceleration "$R" \
    --methods nerp \
    --nerp_iters 1000 \
    --num_shards 8 \
    --shard_id "$SLURM_ARRAY_TASK_ID" \
    --output \
    "outputs/fastmri_nerp1000_tuning30_r${R}_shard${SLURM_ARRAY_TASK_ID}.csv"
