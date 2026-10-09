#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J FULL_COHORT_FINETUNE_OPTIMAL
#SBATCH -o full_finetune_optimal_logs/FULL_COHORT_FINETUNE_OPTIMAL_%A_%a.out
#SBATCH -e full_finetune_optimal_logs/FULL_COHORT_FINETUNE_OPTIMAL_%A_%a.out
#
# Fine-tunes a session-specific model for every session of an experiment/subject,
# one session per array task, starting from the "optimal" full-cohort pretrained
# network (weights only are initialized; optimizer/scheduler start fresh).
#
# Architecture and parameters match run_snn_full_cohort_pretrain_optimal.sbatch:
#   Hidden layers:  512 -> 256 -> 128
#   Threshold:      1.0 (all layers)
#   Reset:          soft (set in CONFIG_YAML)
#   EMA stages:     2
#   tau_syn:        8.0
#
# Pretrained weights are read from
#   snn_checkpoints/<EXPERIMENT>/<SUBJECT>/full_cohort_pretrained_optimal/best_model_weights.pth
# Checkpoints go to
#   snn_checkpoints/<EXPERIMENT>/<SUBJECT>/full_cohort_finetuned_optimal/<SESSION>/
#
# Submit with (full_finetune_optimal_logs/ must exist before submission):
#   mkdir -p full_finetune_optimal_logs
#   N_SESSIONS=$(find /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/indy/mua_8_group -maxdepth 1 -mindepth 1 -type d | wc -l)
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0 run_snn_full_cohort_finetune_array_optimal.sbatch                       # validate one session first
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-$((N_SESSIONS-1)) run_snn_full_cohort_finetune_array_optimal.sbatch   # the rest

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Select Which Experiment/Subject ---
USAGE="sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_snn_full_cohort_finetune_array_optimal.sbatch"
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g.," >&2
    echo "       $USAGE" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g.," >&2
    echo "       $USAGE" >&2
    exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set -- this script must be submitted with --array, e.g.," >&2
    echo "       $USAGE" >&2
    exit 1
fi

# --- Configuration Name (must match the pretraining run) ---
EXP_NAME="optimal"

# --- Customizable Src/Dest Paths ---
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
PRETRAIN_DIR="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_pretrained_${EXP_NAME}"
PRETRAINED_CHECKPOINT="${PRETRAIN_DIR}/best_model_weights.pth"
CHECKPOINTS_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_finetuned_${EXP_NAME}"
CONFIG_YAML="../configs/iaf_soft_reset.yaml"   # soft-reset IAF config whose experiment name is "optimal"

# --- Selected Model Architecture and Parameters (Must Match Pretrained) ---
HIDDEN_DIMS="512 256 128"
THRESHOLDS="1.0 1.0 1.0 1.0"   # one per hidden layer + output layer
TAU_SYN=8.0
TEMPORAL_DECAY_STAGES=2        # EMA staging
BATCH_SIZE=20
FINETUNE_EPOCHS=50
FINETUNE_PATIENCE=10
FINETUNE_SCHEDULER_PATIENCE=10
CHECKPOINT_INTERVAL=10

if [ ! -f "$CONFIG_YAML" ]; then
    echo "ERROR: config $CONFIG_YAML not found." >&2
    exit 1
fi
if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist -- check EXPERIMENT/SUBJECT and that this" >&2
    echo "       subject's mua_8_group datasets have actually been built." >&2
    exit 1
fi
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_CHECKPOINT -- run" >&2
    echo "       run_snn_full_cohort_pretrain_optimal.sbatch for this subject first." >&2
    exit 1
fi
if [ ! -f "${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "WARNING: ${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth not found -- pretraining may" >&2
    echo "         still be running or incomplete; best_model_weights.pth may not be final." >&2
fi

# --- Resolve Session ---
mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)
N_SESSIONS=${#SESSION_DIRS[@]}
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: found zero session directories under $SNN_DATASET_ROOT" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session"
    echo "(only $N_SESSIONS sessions found) -- exiting cleanly."
    exit 0
fi

SESSION_DIR="${SESSION_DIRS[$SLURM_ARRAY_TASK_ID]}"
SESSION_ID="$(basename "$SESSION_DIR")"

if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# --- Check if Final Checkpoint Already Exists. If Not, Resume From Last Checkpoint ---
CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/${SESSION_ID}"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth"
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] ${SESSION_ID}: $FINAL_CHECKPOINT already exists (fine-tuning genuinely"
    echo "       completed) -- delete it (or the whole checkpoint dir) first if you"
    echo "       actually want to redo this."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] ${SESSION_ID}: no final checkpoint, but found an incomplete fine-tuning"
    echo "         run -- resuming from periodic checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
elif [ -f "${CHECKPOINT_DIR}/best_model_weights.pth" ]; then
    echo "[resume] ${SESSION_ID}: no periodic checkpoint found, but best_model_weights.pth"
    echo "         exists -- resuming from that instead."
    RESUME_ARGS+=(--resume "${CHECKPOINT_DIR}/best_model_weights.pth")
else
    echo "[fresh] ${SESSION_ID}: no final, periodic, or best checkpoint found -- starting"
    echo "        fresh from the pretrained model's weights"
fi

# --- Only Initialize From Pretrained Weights if Not Resuming ---
INIT_ARGS=()
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    INIT_ARGS+=(--init-weights-from "$PRETRAINED_CHECKPOINT")
fi

# --- Start Training ---
echo "Array task $SLURM_ARRAY_TASK_ID -> ${EXPERIMENT}/${SUBJECT} session=$SESSION_ID ($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS}) (config=$EXP_NAME)"
echo "  init-weights-from=$PRETRAINED_CHECKPOINT"
echo "  hidden-dims=[$HIDDEN_DIMS] thresholds=[$THRESHOLDS] tau-syn=$TAU_SYN"
echo "  ema-stages=$TEMPORAL_DECAY_STAGES batch-size=$BATCH_SIZE"
echo "  epochs=$FINETUNE_EPOCHS patience=$FINETUNE_PATIENCE checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_bmi.py \
    --data-path "$SESSION_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn multi \
    --min-vmem -1 \
    --use-iaf-squeeze \
    --weight-init kaiming \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --training-mode windowed \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$FINETUNE_EPOCHS" \
    --patience "$FINETUNE_PATIENCE" \
    --scheduler-patience "$FINETUNE_SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type iaf \
    --tau-syn "$TAU_SYN" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
