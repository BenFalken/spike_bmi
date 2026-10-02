#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_FINETUNE
#SBATCH -o snn_logs/SNN_POOLED_FINETUNE_%A_%a.out
#SBATCH -e snn_logs/SNN_POOLED_FINETUNE_%A_%a.out
#
# Stage 2 of 2: fine-tune the pooled pretrained model on each session of one
# subject (array task i = session i). Each run starts from the pretrained
# model's weights (--init-weights-from, fresh optimizer) and uses exactly the
# network and training settings of pretraining (snn_medium_config.sh). With
# TAU_SYN and/or RESET_TYPE set, it fine-tunes the matching pretrained model.
#
# Submit (from this directory), validating one session first:
#   mkdir -p snn_logs
#   N=$(find $DATA_ROOT/snn_datasets/bmi/indy/mua_8_group -mindepth 1 -maxdepth 1 -type d | wc -l)
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0 run_snn_pooled_finetune_array.sbatch
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=1-$((N-1)) run_snn_pooled_finetune_array.sbatch
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/full_cohort_finetuned_medium[_tau_syn_<T>][_soft]/<session>/
# Re-submitting skips finished sessions and resumes interrupted ones.

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g." >&2
    echo "  sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-N $0" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
source ./snn_medium_config.sh

PRETRAINED_CHECKPOINT="${PRETRAIN_DIR}/best_model_weights.pth"
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: $PRETRAINED_CHECKPOINT not found; run run_snn_pooled_pretrain.sbatch" \
         "for ${EXPERIMENT}/${SUBJECT} (TAU_SYN=${TAU_SYN}, RESET_TYPE=${RESET_TYPE}) first." >&2
    exit 1
fi

# --- Resolve session ---
mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)
N_SESSIONS=${#SESSION_DIRS[@]}
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: no session directories under $SNN_DATASET_ROOT" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session ($N_SESSIONS found); exiting."
    exit 0
fi
SESSION_DIR="${SESSION_DIRS[$SLURM_ARRAY_TASK_ID]}"
SESSION_ID="$(basename "$SESSION_DIR")"
if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# --- Skip / resume / start from the pretrained weights ---
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Fine-tuning ${EXPERIMENT}/${SUBJECT}/${SESSION_ID} ($((SLURM_ARRAY_TASK_ID + 1))/${N_SESSIONS}):" \
     "hidden=${HIDDEN_DIMS[*]} thresholds=${THRESHOLDS[*]} tau_syn=${TAU_SYN} reset=${RESET_TYPE} -> ${CHECKPOINT_DIR}"
mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$SESSION_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
