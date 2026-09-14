#!/bin/bash
#SBATCH --job-name=sham256_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=radiology
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_eval_%A_%a.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_eval_%A_%a.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

export PYTHONPATH="code:external/shamaei_original"
export OMP_NUM_THREADS=1

ACC=${ACC:?Must set ACC=4 or ACC=8}
S=${SLURM_ARRAY_TASK_ID}

if [ "${ACC}" -eq 4 ]; then
    CKPT="outputs/shamaei_fastmri256_r4/best.pt"
elif [ "${ACC}" -eq 8 ]; then
    CKPT="outputs/shamaei_fastmri256_r8/best.pt"
else
    echo "ERROR: ACC must be 4 or 8"
    exit 1
fi

OUT="outputs/shamaei_eval_shards/r${ACC}_shard${S}.csv"

echo "============================================================"
echo "SHAMAEI FASTMRI-256 TUNING30 EVAL"
echo "============================================================"
echo "ACC=${ACC}"
echo "SHARD=${S}/4"
echo "CHECKPOINT=${CKPT}"
echo "OUTPUT=${OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

python code/scripts/eval_shamaei_fastmri256_tuning30.py \
    --acceleration "${ACC}" \
    --checkpoint "${CKPT}" \
    --num_shards 4 \
    --shard_id "${S}" \
    --output "${OUT}" \
    --resume

echo
echo "============================================================"
echo "SHAMAEI R${ACC} SHARD ${S} COMPLETE"
echo "============================================================"
