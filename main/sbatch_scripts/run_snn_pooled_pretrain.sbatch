#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_PRETRAIN
#SBATCH -o snn_logs/SNN_POOLED_PRETRAIN_%A_%a.out
#SBATCH -e snn_logs/SNN_POOLED_PRETRAIN_%A_%a.out
#
# Stage 1 of 2: train one SNN on the pooled datasets of a subject's sessions.
# The network and training settings are the "medium" configuration in
# snn_medium_config.sh (hidden 256 -> 128, spike threshold 1.25), shared with
# run_snn_pooled_finetune_array.sbatch. bmi pools the grouped datasets
# (mua_8_group); hkm pools the whole-trial datasets (mua) with batch size 1.
#
#   LOSO=0 (default)  one job: one model pretrained on every session.
#   LOSO=1            an array, task i = session i: a model pretrained on every
#                     session except session i, for fine-tuning on session i.
#                     Session i's data never reaches its pretraining.
#
# Pools are symlinks to the sessions' .pkl files (build_pretraining_pool.py).
#
# Submit (from this directory):
#   mkdir -p snn_logs
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy run_snn_pooled_pretrain.sbatch
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy,TAU_SYN=8.0 run_snn_pooled_pretrain.sbatch
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy,RESET_TYPE=soft run_snn_pooled_pretrain.sbatch
#   # leave-one-session-out, then each fine-tune task starts when its pretraining succeeds:
#   N=$(find $DATA_ROOT/snn_datasets/hkm/jenkins/mua -mindepth 1 -maxdepth 1 -type d | wc -l)
#   PRE=$(sbatch --parsable --time=48:00:00 --export=ALL,EXPERIMENT=hkm,SUBJECT=jenkins,LOSO=1 \
#         --array=0-$((N-1)) run_snn_pooled_pretrain.sbatch)
#   sbatch --dependency=aftercorr:$PRE --time=48:00:00 --export=ALL,EXPERIMENT=hkm,SUBJECT=jenkins,LOSO=1 \
#         --array=0-$((N-1)) run_snn_pooled_finetune_array.sbatch
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/
#   full_cohort_pretrained_medium[_tau_syn_<T>][_soft]/            (LOSO=0)
#   loso_pretrained_medium[_tau_syn_<T>][_soft]/<held-out session>/ (LOSO=1)
# Re-submitting skips a finished run and resumes an interrupted one (for hkm,
# where an epoch over a pool takes hours, expect to re-submit).

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g." >&2
    echo "  sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy $0" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
source ./snn_medium_config.sh

if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist; build the ${EXPERIMENT} SNN datasets first." >&2
    exit 1
fi

# --- Pool and checkpoint directory ---
EXCLUDE_ARGS=()
if [ "$LOSO" = "1" ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: LOSO=1 pretraining is an array (task i holds out session i); submit with --array=0-N" >&2
        exit 1
    fi
    session_dirs
    if [ "${#SESSION_DIRS[@]}" -lt 2 ]; then
        echo "ERROR: leave-one-session-out needs at least 2 sessions under $SNN_DATASET_ROOT" >&2
        exit 1
    fi
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SESSION_DIRS[@]}" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#SESSION_DIRS[@]} found); exiting."
        exit 0
    fi
    HELDOUT_ID="$(basename "${SESSION_DIRS[$SLURM_ARRAY_TASK_ID]}")"
    EXCLUDE_ARGS=(--exclude-session "$HELDOUT_ID")
fi
POOL_DIR="$(pool_dir "${HELDOUT_ID:-}")"
PRETRAIN_DIR="$(pretrain_dir "${HELDOUT_ID:-}")"

if [ -f "${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth exists; delete it to retrain."
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# The full pool is built once; a LOSO pool is rebuilt at every start (symlinks, cheap).
if [ "$LOSO" = "0" ] && [ -n "$(ls -A "$POOL_DIR/train" 2>/dev/null)" ]; then
    echo "[reuse] pool at $POOL_DIR"
else
    rm -rf "$POOL_DIR"
    python3 "$POOL_SCRIPT" --mua-large-root "$SNN_DATASET_ROOT" --output-dir "$POOL_DIR" "${EXCLUDE_ARGS[@]}"
fi

# --- Resume ---
latest_resume_args "$PRETRAIN_DIR"
if [ ${#RESUME_ARGS[@]} -eq 0 ] && [ -f "${PRETRAIN_DIR}/best_model_weights.pth" ]; then
    echo "[resume] ${PRETRAIN_DIR}/best_model_weights.pth (no periodic checkpoint)"
    RESUME_ARGS=(--resume "${PRETRAIN_DIR}/best_model_weights.pth")
fi

echo "Pooled pretraining ${EXPERIMENT}/${SUBJECT}${HELDOUT_ID:+ without ${HELDOUT_ID}}:" \
     "hidden=${HIDDEN_DIMS[*]} thresholds=${THRESHOLDS[*]} tau_syn=${TAU_SYN} reset=${RESET_TYPE}" \
     "batch=${BATCH_SIZE} -> ${PRETRAIN_DIR}"
mkdir -p "$PRETRAIN_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$POOL_DIR" \
    --checkpoint-dir "$PRETRAIN_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
