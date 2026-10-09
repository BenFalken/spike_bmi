#!/bin/bash
#SBATCH --job-name=hkm_nwb_pipeline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
# RE-RUNS: each session is rebuilt from scratch every time (run_nwb_pipeline.sh
# deletes that session's intermediates and previous final outputs first and
# then fails the task if any velocity glitch survives). Sessions never touch
# each other's files -- every intermediate, final and check directory is
# named by session id -- so any subset can safely run at the same time.
#
# SUBJECT selects which subject's sessions form the array (default: nitschke,
# so Jenkins' good data is never rewritten by accident). SUBJECT=all runs
# every subject.
#
# Submit from the directory that holds run_nwb_pipeline.sh and the other
# pipeline scripts:
#   cd /users/bfalkenb/spike_bmi_main/<code dir>
#   mkdir -p logs        # must exist BEFORE submitting, or SLURM cannot open the log files
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/raw/hkm/nitschke -maxdepth 1 -name '*.nwb' | wc -l)
#   sbatch --array=0-$((N-1)) run_hkm_nwb_pipeline_array.sbatch
# Other subject:  sbatch --export=ALL,SUBJECT=jenkins --array=0-3 ...
# Throttle (optional): --array=0-5%3 runs at most 3 sessions at once.

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

# SLURM runs a COPY of this file from its spool directory, so BASH_SOURCE
# points there, not at your code. The code directory is therefore the
# directory sbatch was submitted from (or CODE_DIR if you export it).
SCRIPT_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NWB_PIPELINE_SCRIPT="$SCRIPT_DIR/run_nwb_pipeline.sh"
if [ ! -f "$NWB_PIPELINE_SCRIPT" ]; then
    echo "ERROR: run_nwb_pipeline.sh not found at $NWB_PIPELINE_SCRIPT -- " >&2
    echo "       submit from the directory that contains it (or export CODE_DIR)." >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set -- submit with sbatch --array=0-N" >&2
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
SUBJECT="${SUBJECT:-nitschke}"
if [ "$SUBJECT" = "all" ]; then
    mapfile -t NWB_FILES < <(find "$RAW_DATA_DIR" -mindepth 2 -maxdepth 2 -name '*.nwb' | sort)
else
    mapfile -t NWB_FILES < <(find "$RAW_DATA_DIR/$SUBJECT" -mindepth 1 -maxdepth 1 -name '*.nwb' 2>/dev/null | sort)
fi

if [ "${#NWB_FILES[@]}" -eq 0 ]; then
    echo "ERROR: found zero .nwb files two levels under $RAW_DATA_DIR " >&2
    echo "(SUBJECT=$SUBJECT; expected {subject}/{session}.nwb) -- check RAW_DATA_DIR matches the " >&2
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

echo "Subject filter: $SUBJECT   sessions found: ${#NWB_FILES[@]}"
echo "Array task $SLURM_ARRAY_TASK_ID / $((${#NWB_FILES[@]} - 1)) -> $NWB_FILE"
echo "Code dir: $SCRIPT_DIR"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

bash "$NWB_PIPELINE_SCRIPT" \
    --nwb-path "$NWB_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --snn-dataset-root "$SNN_DATASET_ROOT" \
    --test-frac "$TEST_FRAC" \
    --feature "$FEATURE" \
    --method "$METHOD" \
    --cleanup-intermediate
