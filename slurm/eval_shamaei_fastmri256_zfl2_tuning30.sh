#!/bin/bash
#SBATCH --job-name=sham_zfl2_eval
#SBATCH --array=4,8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=radiology
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham_zfl2_eval_%A_%a.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham_zfl2_eval_%A_%a.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source \
  /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh

conda activate med

export PYTHONPATH="code:external/shamaei_original"
export OMP_NUM_THREADS=1

ACC="${SLURM_ARRAY_TASK_ID}"

CHECKPOINT="outputs/shamaei_fastmri256_zfl2_r${ACC}/best.pt"

OUTPUT="outputs/shamaei_fastmri256_zfl2_tuning30_r${ACC}.csv"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint: ${CHECKPOINT}"
    exit 1
fi

if [ -f "${OUTPUT}" ]; then
    echo "ERROR: output already exists:"
    echo "${OUTPUT}"
    echo "Move/remove it explicitly before rerunning."
    exit 1
fi

echo "============================================================"
echo "Corrected Shamaei tuning30 evaluation"
echo "Acceleration: R${ACC}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Calibration: ZF-RSS L2 norm matching"
echo "Output: ${OUTPUT}"
echo "============================================================"

python \
  code/scripts/eval_shamaei_fastmri256_tuning30.py \
  --acceleration "${ACC}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}"
