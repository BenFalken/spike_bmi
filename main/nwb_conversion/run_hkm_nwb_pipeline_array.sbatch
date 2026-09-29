#!/bin/bash
#SBATCH --job-name=hkm_nwb_pipeline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#
# Runs run_nwb_pipeline.sh once per session, in parallel, across every
# subject under the "hkm" experiment's raw NWB root. Mirrors
# run_pipeline_array.sbatch's own glob/array-resolution pattern exactly
# (deterministic two-levels-deep discovery, sorted so every task agrees
# on ordering, subjects naturally interleaved rather than one subject's
# sessions all finishing before the next starts) -- same structure,
# applied to .nwb files and run_nwb_pipeline.sh instead of .mat files
# and single_subject_pipeline.py.
#
# This code is safe to re-run from any failure/stoppage point: each
# session's own conversion writes into its own, session-named
# subdirectory (see run_nwb_pipeline.sh's own stage-skip checks), so a
# resubmission does not redo a session that already finished.
#
# Submit with:
#   mkdir -p logs
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/raw/hkm -mindepth 2 -maxdepth 2 -name '*.nwb' | wc -l)
#   sbatch --array=0-$((N-1)) run_hkm_nwb_pipeline_array.sbatch

set -eo pipefail
# NOTE: -u (errexit on unbound variable) deliberately left off here -- some
# venv activate scripts reference variables that are unset on a fresh,
# non-interactive shell, and -u would abort on THAT instead of anything
# related to the actual job.

# --- Environment ---
# run_nwb_pipeline.sh's own stages (convert_nwb_trials_to_raw_h5.py,
# make_dataset.py, make_snn_dataset_whole_trial.py) are the same ANN-side
# stack single_subject_pipeline.py uses (pynwb/h5py/numpy, not torch/
# sinabs) -- EDIT if pynwb specifically lives in a separate env from
# ann-env on your setup; not verified here.
deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# --- CPU-only thread capping ---
# Multiple array tasks run concurrently on the same shared node -- see
# the earlier, confirmed discussion for why this matters specifically
# for array jobs (uncapped threads means every concurrent task fights
# every other task for every core). Set BEFORE any python starts.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

# --- Fixed settings -- YOUR actual paths/options, matching the single-
# session command you already ran successfully. EDIT as needed. ---
RAW_DATA_DIR="/users/bfalkenb/scratch/bfalkenb/data/raw/hkm"
# Intermediate, DISPOSABLE staging only -- raw + ANN-windowed per-trial
# files land here temporarily, then get deleted by --cleanup-intermediate
# below once the two real destinations (DATASET_ROOT/SNN_DATASET_ROOT)
# have what they need. Under the same .../data/ root as everything else
# in this project, clearly separate from "dataset"/"snn_datasets" (the
# REAL destinations) and from "hkm_purgatory" (an earlier, now-abandoned
# location that held FINAL outputs incorrectly -- this is intermediate
# staging only, nothing here is ever a final destination).
OUTPUT_ROOT="/users/bfalkenb/scratch/bfalkenb/data/hkm_intermediate"
DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/dataset"
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets"
TEST_FRAC=0.1
FEATURE=mua
METHOD=binning

# run_nwb_pipeline.sh MUST be resolvable from wherever this job's cwd
# ends up (SLURM's default: wherever `sbatch` was run from) -- same
# resolution issue diagnosed earlier for single_subject_pipeline.py's
# own subprocess calls. Anchored here to this array script's own
# location rather than assumed to be a sibling of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NWB_PIPELINE_SCRIPT="./run_nwb_pipeline.sh"
if [ ! -f "$NWB_PIPELINE_SCRIPT" ]; then
    echo "ERROR: run_nwb_pipeline.sh not found at $NWB_PIPELINE_SCRIPT -- " >&2
    echo "       expected it as a sibling of this array script." >&2
    exit 1
fi

# --- Resolve this array task's session ---
# mindepth/maxdepth 2 -- exactly {subject}/{session}.nwb, not deeper or
# shallower (matches raw/hkm/jenkins/*.nwb, raw/hkm/nitschke/*.nwb).
# Deterministically sorted, so every task's glob agrees on ordering no
# matter what order they each happen to actually run in; also means
# jenkins' and nitschke's sessions are naturally interleaved (sorted by
# subject name first), so an early --array range doesn't accidentally
# finish one subject entirely before starting the next.
mapfile -t NWB_FILES < <(find "$RAW_DATA_DIR" -mindepth 2 -maxdepth 2 -name '*.nwb' | sort)

if [ "${#NWB_FILES[@]}" -eq 0 ]; then
    echo "ERROR: found zero .nwb files two levels under $RAW_DATA_DIR " >&2
    echo "(expected {subject}/{session}.nwb) -- check RAW_DATA_DIR matches the " >&2
    echo "real raw data layout, and that this job's working directory is what " >&2
    echo "you expect (SLURM's default is the directory 'sbatch' was run from)." >&2
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#NWB_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session "
    echo "(only ${#NWB_FILES[@]} found under $RAW_DATA_DIR) -- exiting cleanly."
    exit 0
fi

NWB_FILE="${NWB_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID / $((${#NWB_FILES[@]} - 1)) -> $NWB_FILE"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

"$NWB_PIPELINE_SCRIPT" \
    --nwb-path "$NWB_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --snn-dataset-root "$SNN_DATASET_ROOT" \
    --test-frac "$TEST_FRAC" \
    --feature "$FEATURE" \
    --method "$METHOD" \
    --cleanup-intermediate
