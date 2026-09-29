#!/bin/bash
#SBATCH --job-name=bmi_pipeline
#SBATCH --output=bmi_logs/%x_%A_%a.out
#SBATCH --error=bmi_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# Runs the entire data processing/storage/decoding pipeline for
# ANN/KF/WF. Stores the binned neural and hand velocity datasets,
# trains/optimizes all decoders, and stores metrics on accuracy,
# power consumption, training time and latency.
#
# This code is safe to re-run from any failure/stoppage point.
#
# Submit with:
#   mkdir -p bmi_logs
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/raw/bmi -mindepth 2 -maxdepth 2 -name '*.mat' | wc -l)
#   sbatch --array=0-46 run_bmi_subject_pipeline_array.sbatch
#

set -eo pipefail

# --- Activate Environment ---
deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# --- CPU-only Thread Capping ---
# TensorFlow/PyTorch/BLAS each try to claim every core on the node by
# default. With multiple array tasks running concurrently on a shared
# node, uncapped threads means every task fights every other task for
# every core. Cap each task to exactly what SLURM allocated it, set before
# python starts (TF/PyTorch read these at import time).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

# --- Resolve Session ---
RAW_DATA_DIR="/users/bfalkenb/scratch/bfalkenb/data/raw/bmi"

mapfile -t RAW_FILES < <(find "$RAW_DATA_DIR" -mindepth 2 -maxdepth 2 -name '*.mat' | sort)

if [ "${#RAW_FILES[@]}" -eq 0 ]; then
    echo "ERROR: found zero .mat files two levels under $RAW_DATA_DIR " >&2
    echo "(expected {subject}/{raw_stem}.mat)" >&2
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#RAW_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session "
    echo "(only ${#RAW_FILES[@]} found under $RAW_DATA_DIR) -- exiting cleanly."
    exit 0
fi

RAW_FILE="${RAW_FILES[$SLURM_ARRAY_TASK_ID]}"
RAW_STEM="$(basename "$RAW_FILE" .mat)"

# --- Run Pipeline ---
echo "Array task $SLURM_ARRAY_TASK_ID / $((${#RAW_FILES[@]} - 1)) -> $RAW_STEM"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

python3 ../preprocessing_training/single_subject_pipeline.py --raw_stem "$RAW_STEM"
