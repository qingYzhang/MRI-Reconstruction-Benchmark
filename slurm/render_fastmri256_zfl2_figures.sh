#!/bin/bash
#SBATCH --job-name=fmri_fig
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=radiology
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/fmri_fig_%A_%a.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/fmri_fig_%A_%a.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source \
  /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh

conda activate med

export PYTHONPATH="code:external/laps_original/src:external/shamaei_original"
export OMP_NUM_THREADS=1

ACC=${ACC:?Must set ACC=4 or ACC=8}

NUM_SHARDS=8
SHARD_ID="${SLURM_ARRAY_TASK_ID}"

CHECKPOINT="outputs/shamaei_fastmri256_zfl2_r${ACC}/best.pt"

OUT="outputs/reconstruction_figures/advisor_r${ACC}/shard_${SHARD_ID}"

mkdir -p "${OUT}"

python \
  code/scripts/render_fastmri256_reconstruction_figures.py \
  --acceleration "${ACC}" \
  --checkpoint "${CHECKPOINT}" \
  --split_json \
    outputs/shamaei_fastmri256_split.json \
  --output_dir "${OUT}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --resume
