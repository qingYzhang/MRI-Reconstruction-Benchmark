#!/bin/bash
#SBATCH --job-name=sham256_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gl40s_short
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_train_%j.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_train_%j.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh
conda activate med

export PYTHONPATH="code:external/shamaei_original"
export OMP_NUM_THREADS=1

ACC=${ACC:?Must set ACC=4 or ACC=8}

OUT="outputs/shamaei_fastmri256_r${ACC}"

mkdir -p "${OUT}"

RESUME_ARGS=()

if [ -f "${OUT}/last.pt" ]; then
    echo "Resuming from ${OUT}/last.pt"
    RESUME_ARGS=(
        --resume
        "${OUT}/last.pt"
    )
else
    echo "Starting fresh formal R${ACC} training"
fi

torchrun \
    --standalone \
    --nproc_per_node=4 \
    code/scripts/train_shamaei_fastmri256.py \
    --acceleration ${ACC} \
    --output_dir "${OUT}" \
    --max_epochs 200 \
    --learning_rate 0.001 \
    --accum_steps 16 \
    --early_stop_patience 10 \
    --num_workers 2 \
    --expected_world_size 4 \
    "${RESUME_ARGS[@]}"