#!/bin/bash
#SBATCH --job-name=hkm_nwb_pipeline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#
# Build the HKM datasets: run_nwb_pipeline.sh once per NWB session, one array
# task per session. Sessions are {RAW_DATA_DIR}/<subject>/*.nwb, sorted;
# SUBJECT=<name> limits the array to one subject (default: every subject).
#
# Each task rebuilds its session from scratch (see run_nwb_pipeline.sh), so
# re-submitting any subset is safe, and no two sessions share files.
#
# Submit from this directory (nwb_conversion/; logs/ must exist first):
#   mkdir -p logs
#   N=$(find $DATA_ROOT/raw/hkm -mindepth 2 -maxdepth 2 -name '*.nwb' | wc -l)
#   sbatch --array=0-$((N-1)) run_hkm_nwb_pipeline_array.sbatch
#   sbatch --export=ALL,SUBJECT=nitschke --array=0-$((N_NITSCHKE-1)) run_hkm_nwb_pipeline_array.sbatch

set -eo pipefail

deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
RAW_DATA_DIR="${DATA_ROOT}/raw/hkm"
OUTPUT_ROOT="${DATA_ROOT}/hkm_intermediate"     # per-trial intermediates, deleted after each session
DATASET_ROOT="${DATA_ROOT}/dataset"
SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets"
TEST_FRAC=0.1
FEATURE=mua
METHOD=binning

# Slurm runs a copy of this file, so find run_nwb_pipeline.sh from the
# submission directory (or CODE_DIR).
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$CODE_DIR/run_nwb_pipeline.sh" ]; then
    echo "ERROR: no run_nwb_pipeline.sh in $CODE_DIR; submit from nwb_conversion/ (or set CODE_DIR)" >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set; submit with sbatch --array=0-N" >&2
    exit 1
fi

if [ -n "${SUBJECT:-}" ]; then
    mapfile -t NWB_FILES < <(find "$RAW_DATA_DIR/$SUBJECT" -mindepth 1 -maxdepth 1 -name '*.nwb' 2>/dev/null | sort)
else
    mapfile -t NWB_FILES < <(find "$RAW_DATA_DIR" -mindepth 2 -maxdepth 2 -name '*.nwb' | sort)
fi
if [ "${#NWB_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no <subject>/*.nwb under $RAW_DATA_DIR${SUBJECT:+ for SUBJECT=$SUBJECT}" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#NWB_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#NWB_FILES[@]} found); exiting."
    exit 0
fi
NWB_FILE="${NWB_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID / $((${#NWB_FILES[@]} - 1)) -> $NWB_FILE"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

bash "$CODE_DIR/run_nwb_pipeline.sh" \
    --nwb-path "$NWB_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --snn-dataset-root "$SNN_DATASET_ROOT" \
    --test-frac "$TEST_FRAC" \
    --feature "$FEATURE" \
    --method "$METHOD" \
    --cleanup-intermediate
