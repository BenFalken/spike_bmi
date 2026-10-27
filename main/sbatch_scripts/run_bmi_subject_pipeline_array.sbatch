#!/bin/bash
#SBATCH --job-name=bmi_pipeline
#SBATCH --output=bmi_logs/%x_%A_%a.out
#SBATCH --error=bmi_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# bmi sessions, one array task per session ({DATA_ROOT}/raw/bmi/<subject>/<session>.mat,
# sorted): single_subject_pipeline.py downloads and processes the session,
# builds its ANN and SNN datasets and trains KF, WF, LSTM and QRNN on them.
#
#   SUBJECT=<name>   only that subject's sessions (default: every subject)
#   STAGE=datasets   build the datasets only (SNN training can then start while
#                    the decoders train in a second run)
#   OVERWRITE=1      recompute every stage
#
# Every stage skips outputs that already exist, so re-submitting resumes.
#
# Submit from this directory:
#   mkdir -p bmi_logs
#   N=$(find $DATA_ROOT/raw/bmi/indy -maxdepth 1 -name '*.mat' | wc -l)
#   sbatch --export=ALL,SUBJECT=indy --array=0-$((N-1)) run_bmi_subject_pipeline_array.sbatch

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
RAW_DATA_DIR="${DATA_ROOT}/raw/bmi"

if [ -n "${SUBJECT:-}" ]; then
    mapfile -t RAW_FILES < <(find "$RAW_DATA_DIR/$SUBJECT" -mindepth 1 -maxdepth 1 -name '*.mat' 2>/dev/null | sort)
else
    mapfile -t RAW_FILES < <(find "$RAW_DATA_DIR" -mindepth 2 -maxdepth 2 -name '*.mat' | sort)
fi
if [ "${#RAW_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no <subject>/<session>.mat under $RAW_DATA_DIR${SUBJECT:+ for SUBJECT=$SUBJECT}" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#RAW_FILES[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#RAW_FILES[@]} found); exiting."
    exit 0
fi
RAW_STEM="$(basename "${RAW_FILES[$SLURM_ARRAY_TASK_ID]}" .mat)"

echo "Array task $SLURM_ARRAY_TASK_ID / $((${#RAW_FILES[@]} - 1)) -> $RAW_STEM${STAGE:+ (stage: $STAGE)}"
echo "Host: $(hostname), CPUs allocated: $SLURM_CPUS_PER_TASK"

ARGS=()
[ "${STAGE:-}" = "datasets" ] && ARGS+=(--datasets_only)
[ "${OVERWRITE:-0}" = "1" ] && ARGS+=(--overwrite)
python3 ../preprocessing_training/single_subject_pipeline.py --raw_stem "$RAW_STEM" "${ARGS[@]}"
