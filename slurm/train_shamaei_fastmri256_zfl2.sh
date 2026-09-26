#!/bin/bash
#SBATCH --job-name=sham256_zfl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu_long
#SBATCH --output=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_zfl2_%j.out
#SBATCH --error=/gpfs/data/chopralab/zhangs18/fmri/slurm_logs/sham256_zfl2_%j.err

set -euo pipefail

cd /gpfs/data/chopralab/zhangs18/fmri

source \
  /gpfs/scratch/zhangs18/miniconda3/etc/profile.d/conda.sh

conda activate med

export PYTHONPATH="code:external/shamaei_original"
export OMP_NUM_THREADS=1

ACC=${ACC:?Must set ACC=4 or ACC=8}

if [ "${ACC}" = "4" ]; then
    #
    # Preserve the exact optimization setting used by
    # the original R4 experiment so calibration is the
    # only methodological change.
    #
    ACCUM_STEPS=8

elif [ "${ACC}" = "8" ]; then
    #
    # Preserve the exact optimization setting used by
    # the original R8 experiment.
    #
    ACCUM_STEPS=16

else
    echo "ACC must be 4 or 8"
    exit 1
fi

OUT="outputs/shamaei_fastmri256_zfl2_r${ACC}"

mkdir -p "${OUT}"
mkdir -p slurm_logs

RESUME_ARGS=()

if [ -f "${OUT}/last.pt" ]; then

    echo "Resuming corrected R${ACC} training from:"
    echo "${OUT}/last.pt"

    RESUME_ARGS=(
        --resume
        "${OUT}/last.pt"
    )

else

    echo "Starting fresh corrected R${ACC} training"
    echo "Output: ${OUT}"
    echo "Calibration: ZF-RSS L2 norm matching"
    echo "Accumulation steps: ${ACCUM_STEPS}"

fi

torchrun \
    --standalone \
    --nproc_per_node=4 \
    code/scripts/train_shamaei_fastmri256.py \
    --acceleration "${ACC}" \
    --output_dir "${OUT}" \
    --max_epochs 200 \
    --learning_rate 0.001 \
    --accum_steps "${ACCUM_STEPS}" \
    --early_stop_patience 10 \
    --num_workers 2 \
    --expected_world_size 4 \
    "${RESUME_ARGS[@]}"
