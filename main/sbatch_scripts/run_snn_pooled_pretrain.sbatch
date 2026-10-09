#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_PRETRAIN
#SBATCH -o snn_logs/SNN_POOLED_PRETRAIN_%A_%a.out
#SBATCH -e snn_logs/SNN_POOLED_PRETRAIN_%A_%a.out
#
# Stage 1 of 2 of the PyTorch SNN: pretrain the "pooled" network
# (snn_config.sh: 512 -> 256 -> 128, tau_syn 2-16) on the pooled datasets of
# one subject's sessions; run_snn_pooled_finetune_array.sbatch then fine-tunes
# it per session.
#
#   LOSO=1 (default)  an array, task i = session i: pretrained on every session
#                     except session i, so session i's data never reaches it.
#   LOSO=0            one job: pretrained on every session.
#
# Pools are symlinks to the sessions' .pkl files (build_pretraining_pool.py).
#
# Submit (from this directory), each fine-tune task starting when its
# pretraining succeeds:
#   mkdir -p snn_logs
#   N=$(find $DATA_ROOT/snn_datasets/bmi/indy/mua_8_group -mindepth 1 -maxdepth 1 -type d | wc -l)
#   PRE=$(sbatch --parsable --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-$((N-1)) run_snn_pooled_pretrain.sbatch)
#   sbatch --dependency=aftercorr:$PRE --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-$((N-1)) \
#       run_snn_pooled_finetune_array.sbatch
#   (hkm: sessions under snn_datasets/hkm/<subject>/mua)
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/loso_pretrained/<held-out session>/
#              (LOSO=0: .../full_cohort_pretrained/)
# Re-submitting skips a finished run and resumes an interrupted one (for hkm,
# where an epoch over a pool takes hours, expect to re-submit).

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g. sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-N $0" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
source ./snn_config.sh
select_model pooled

EXCLUDE_ARGS=()
if [ "$LOSO" = "1" ]; then
    array_session
    if [ "${#SESSION_DIRS[@]}" -lt 2 ]; then
        echo "ERROR: leave-one-session-out needs at least 2 sessions under $SNN_DATASET_ROOT" >&2
        exit 1
    fi
    HELDOUT_ID="$SESSION_ID"
    EXCLUDE_ARGS=(--exclude-session "$HELDOUT_ID")
fi
POOL_DIR="$(pool_dir "${HELDOUT_ID:-}")"
PRETRAIN_DIR="$(pretrain_dir "${HELDOUT_ID:-}")"

if [ -f "${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth exists; delete it to retrain."
    exit 0
fi
activate_env

# The full pool is built once; a LOSO pool is rebuilt at every start (symlinks, cheap).
if [ "$LOSO" = "0" ] && [ -n "$(ls -A "$POOL_DIR/train" 2>/dev/null)" ]; then
    echo "[reuse] pool at $POOL_DIR"
else
    rm -rf "$POOL_DIR"
    python3 "$POOL_SCRIPT" --mua-large-root "$SNN_DATASET_ROOT" --output-dir "$POOL_DIR" "${EXCLUDE_ARGS[@]}"
fi

latest_resume_args "$PRETRAIN_DIR"
echo "Pretraining ${EXPERIMENT}/${SUBJECT}${HELDOUT_ID:+ without ${HELDOUT_ID}} (${MODEL_DESC}) -> ${PRETRAIN_DIR}"
mkdir -p "$PRETRAIN_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$POOL_DIR" \
    --checkpoint-dir "$PRETRAIN_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
