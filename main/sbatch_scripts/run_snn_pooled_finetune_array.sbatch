#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_FINETUNE
#SBATCH -o snn_logs/SNN_POOLED_FINETUNE_%A_%a.out
#SBATCH -e snn_logs/SNN_POOLED_FINETUNE_%A_%a.out
#
# Stage 2 of 2 of the PyTorch SNN: fine-tune the pretrained "pooled" network
# on each session of one subject (array task i = session i). Each run starts
# from the pretrained weights (--init-weights-from, fresh optimizer) with the
# same network and training settings (snn_config.sh); with LOSO=1 (default),
# session i starts from the model pretrained without it. See
# run_snn_pooled_pretrain.sbatch for how to chain the two.
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/loso_finetuned/<session>/
#              (LOSO=0: .../full_cohort_finetuned/<session>/)
# These are the 'snn' checkpoints of run_inference.sbatch.
# Re-submitting skips finished sessions and resumes interrupted ones.

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g. sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-N $0" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
source ./snn_config.sh
select_model pooled
array_session

PRETRAINED_CHECKPOINT="$(pretrain_dir "$SESSION_ID")/best_model_weights.pth"
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: $PRETRAINED_CHECKPOINT not found; run run_snn_pooled_pretrain.sbatch for" \
         "${EXPERIMENT}/${SUBJECT} (LOSO=${LOSO}) first." >&2
    exit 1
fi

CHECKPOINT_DIR="${FINETUNE_ROOT}/${SESSION_ID}"
if [ -f "${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${SESSION_ID}: ${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth exists; delete it to redo."
    exit 0
fi
latest_resume_args "$CHECKPOINT_DIR"
INIT_ARGS=()
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    echo "[fresh] ${SESSION_ID}: starting from $PRETRAINED_CHECKPOINT"
    INIT_ARGS=(--init-weights-from "$PRETRAINED_CHECKPOINT")
fi
activate_env

echo "Fine-tuning ${EXPERIMENT}/${SUBJECT}/${SESSION_ID} ($((SLURM_ARRAY_TASK_ID + 1))/${#SESSION_DIRS[@]})" \
     "(${MODEL_DESC}) -> ${CHECKPOINT_DIR}"
mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$SESSION_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
