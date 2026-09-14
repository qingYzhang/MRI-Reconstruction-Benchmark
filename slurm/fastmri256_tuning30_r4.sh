#!/bin/bash
#SBATCH --job-name=fmri256_r4
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=radiology
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/fmri256_r4_%A_%a.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/fmri256_r4_%A_%a.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

S=${SLURM_ARRAY_TASK_ID}

echo "========================================"
echo "fastMRI-256 tuning30 R4"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array ID:  ${S}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "========================================"

export PYTHONPATH="code:external/laps_original/src:external/laps_original/submodules/diffusers/src"

python code/scripts/eval_fastmri256_prior_free.py \
    --subset tuning30 \
    --acceleration 4 \
    --methods zf cg lacs nerp caps \
    --num_shards 8 \
    --shard_id ${S} \
    --output outputs/fastmri256_tuning30_r4_shard${S}.csv \
    --resume \
    2>&1 | tee outputs/fastmri256_tuning30_r4_shard${S}.log

echo "========================================"
echo "R4 shard ${S} DONE"
echo "End: $(date)"
echo "========================================"
