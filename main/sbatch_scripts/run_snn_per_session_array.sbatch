#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_PER_SESSION
#SBATCH -o snn_logs/SNN_PER_SESSION_%A_%a.out
#SBATCH -e snn_logs/SNN_PER_SESSION_%A_%a.out
#
# The Speck SNN: the "per_session" network (snn_config.sh: 256 -> 128 -> 64,
# no tau_syn, so it deploys to the chip) trained from scratch on each session
# of one subject (array task i = session i).
#
# Submit (from this directory):
#   mkdir -p snn_logs
#   N=$(find $DATA_ROOT/snn_datasets/bmi/indy/mua_8_group -mindepth 1 -maxdepth 1 -type d | wc -l)
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-$((N-1)) run_snn_per_session_array.sbatch
#   (hkm: sessions under snn_datasets/hkm/<subject>/mua)
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/per_session/<session>/
# These are the 'speck' checkpoints of run_inference.sbatch.
# Re-submitting skips finished sessions and resumes interrupted ones.

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g. sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-N $0" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
source ./snn_config.sh
select_model per_session
array_session

CHECKPOINT_DIR="${PER_SESSION_ROOT}/${SESSION_ID}"
if [ -f "${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${SESSION_ID}: ${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth exists; delete it to redo."
    exit 0
fi
latest_resume_args "$CHECKPOINT_DIR"
activate_env

echo "Per-session training ${EXPERIMENT}/${SUBJECT}/${SESSION_ID} ($((SLURM_ARRAY_TASK_ID + 1))/${#SESSION_DIRS[@]})" \
     "(${MODEL_DESC}) -> ${CHECKPOINT_DIR}"
mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$SESSION_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
