#!/bin/bash
#SBATCH --job-name=hkm_pipeline
#SBATCH --output=hkm_logs/%x_%A_%a.out
#SBATCH --error=hkm_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#
# HKM equivalent of run_bmi_subject_pipeline_array.sbatch -- same
# structure, adapted for two real differences: (1) HKM sessions are
# already-converted _binning.h5 files (run_nwb_pipeline.sh's own
# output), not raw .mat files needing --raw_stem alone, so this calls
# single_subject_pipeline.py with --nwb_dataset_filepath (routing to
# process_nwb_session(), not process_session() -- see that function's
# own docstring for why the two aren't interchangeable). (2) --mem is
# raised from 32G to 64G -- confirmed directly that at least one real
# HKM session (n_train=964481, roughly 30x a typical BMI session) was
# large enough to SIGKILL eval_wf_decoder.py's tap-delay feature matrix
# under the BMI script's own 32G.
#
# This code is safe to re-run from any failure/stoppage point -- same
# as the BMI script, single_subject_pipeline.py's own internal per-stage
# skip logic handles this, nothing bash-level needed here.
#
# Submit with:
#   mkdir -p hkm_logs
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/dataset/hkm -mindepth 2 -maxdepth 2 -name '*_binning.h5' | wc -l)
#   sbatch --array=0-10 run_hkm_subject_pipeline_array.sbatch
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
# {BMI_DATA_ROOT}/dataset/hkm/{subject}/{session}_binning.h5 -- matches
# the REAL, confirmed location run_nwb_pipeline.sh's output actually
# landed at (not assumed from that script's own source alone). mindepth/
# maxdepth 2 -- exactly {subject}/{session}_binning.h5, no feature or
# group-size subdirectory in between (unlike the BMI/.mat convention).
DATASET_DIR="/users/bfalkenb/scratch/bfalkenb/data/dataset/hkm"

mapfile -t H5_FILES < <(find "$DATASET_DIR" -mindepth 2 -maxdepth 2 -name '*_binning.h5' | sort)

if [ "${#H5_FILES[@]}" -eq 0 ]; then
    echo "ERROR: found zero *_binning.h5 files two levels under $DATASET_DIR " >&2
    echo "(expected {subject}/{session}_binning.h5)" >&2
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#H5_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session "
    echo "(only ${#H5_FILES[@]} found under $DATASET_DIR) -- exiting cleanly."
    exit 0
fi

H5_FILE="${H5_FILES[$SLURM_ARRAY_TASK_ID]}"
# Strips the _binning.h5 suffix to recover the session ID -- assumes
# METHOD=binning (this project's established default, and what
# run_nwb_pipeline.sh was actually run with here); a session converted
# with a different --method would need this adjusted to match.
RAW_STEM="$(basename "$H5_FILE" _binning.h5)"

# --- Run Pipeline ---
echo "Array task $SLURM_ARRAY_TASK_ID / $((${#H5_FILES[@]} - 1)) -> $RAW_STEM"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

python3 ../preprocessing_training/single_subject_pipeline.py \
    --raw_stem "$RAW_STEM" \
    --nwb_dataset_filepath "$H5_FILE"
