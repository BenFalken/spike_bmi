#!/bin/bash
#SBATCH --job-name=hkm_pipeline
#SBATCH --output=hkm_logs/%x_%A_%a.out
#SBATCH --error=hkm_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#
# HKM counterpart of run_bmi_subject_pipeline_array.sbatch: trains and
# evaluates KF, WF, LSTM and QRNN on every HKM session, one array task per
# session. The datasets come from nwb_conversion/run_hkm_nwb_pipeline_array.sbatch
# ({DATA_ROOT}/dataset/hkm/<subject>/mua/<session>_binning.h5), so this runs
# single_subject_pipeline.py --nwb_dataset_filepath (process_nwb_session()),
# which takes the train/test boundary from the dataset. HKM sessions hold up
# to ~30x the rows of a bmi session, hence the memory (WF's tap-delay matrix).
#
# Every stage skips outputs that already exist, so a re-submission resumes.
# A session whose model bundles are older than its dataset (rebuilt since)
# is retrained from scratch; OVERWRITE=1 retrains every session.
# SUBJECT=<name> limits the array to one subject (default: every subject).
#
# Submit from this directory:
#   mkdir -p hkm_logs
#   N=$(find $DATA_ROOT/dataset/hkm/jenkins/mua -name '*_binning.h5' | wc -l)
#   sbatch --export=ALL,SUBJECT=jenkins --array=0-$((N-1)) run_hkm_subject_pipeline_array.sbatch

set -eo pipefail

deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# Cap each task's threads to its allocation (TF/PyTorch read these at import).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
export BMI_DATA_ROOT="$DATA_ROOT"
DATASET_DIR="${DATA_ROOT}/dataset/hkm"

# {subject}/mua/{session}_binning.h5
if [ -n "${SUBJECT:-}" ]; then
    mapfile -t H5_FILES < <(find "$DATASET_DIR/$SUBJECT/mua" -mindepth 1 -maxdepth 1 -name '*_binning.h5' 2>/dev/null | sort)
else
    mapfile -t H5_FILES < <(find "$DATASET_DIR" -mindepth 3 -maxdepth 3 -name '*_binning.h5' | sort)
fi
if [ "${#H5_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no <subject>/mua/<session>_binning.h5 under $DATASET_DIR${SUBJECT:+ for SUBJECT=$SUBJECT}" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#H5_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#H5_FILES[@]} found); exiting."
    exit 0
fi
H5_FILE="${H5_FILES[$SLURM_ARRAY_TASK_ID]}"
RAW_STEM="$(basename "$H5_FILE" _binning.h5)"

echo "Array task $SLURM_ARRAY_TASK_ID / $((${#H5_FILES[@]} - 1)) -> $RAW_STEM"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

OVERWRITE_ARGS=()
MODEL_DIR="${DATA_ROOT}/results/model_cache/hkm/$(basename "$(dirname "$(dirname "$H5_FILE")")")/${RAW_STEM}/mua"
if [ "${OVERWRITE:-0}" = "1" ]; then
    OVERWRITE_ARGS=(--overwrite)
elif [ -n "$(find "$MODEL_DIR" -name '*_config.json' ! -newer "$H5_FILE" 2>/dev/null)" ]; then
    echo "[retrain] model bundles in $MODEL_DIR are older than $H5_FILE"
    OVERWRITE_ARGS=(--overwrite)
fi

python3 ../preprocessing_training/single_subject_pipeline.py \
    --raw_stem "$RAW_STEM" \
    --nwb_dataset_filepath "$H5_FILE" \
    "${OVERWRITE_ARGS[@]}"
