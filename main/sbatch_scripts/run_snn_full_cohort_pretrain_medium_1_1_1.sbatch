#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J FULL_COHORT_PRETRAIN
#SBATCH -o full_pretrain_medium_1_1_1_logs/FULL_COHORT_PRETRAIN_%j.out
#SBATCH -e full_pretrain_medium_1_1_1_logs/FULL_COHORT_PRETRAIN_%j.out
#
# Pools all sessions of an experiment together and trains a model on the full dataset.
# Uses optimized parameters found in previous sweeps.
#
# Merges train and test data in different configurations:
#   - TRAIN: every session's own train trials are already the same
#     length (group_size=8). Pooling introduces no length
#     mismatch, enabling training in parallel (batches of size >1).
#   - TEST: each session's test split is its own single, whole,
#     unchunked trial. Sessions have different recording durations,
#     so a pooled test set has genuinely different
#     trial lengths across sessions. Batch size is now forced to 1 in
#     instances where trial length differa.
#
# Checkpoint goes to checkpoints/bmi/mua_8_group_all_sessions_pooled/
#
# Submit with:
#   mkdir -p full_pretrain_medium_1_1_1_logs
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=indy run_snn_full_cohort_pretrain_medium_1_1_1.sbatch
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=loco run_snn_full_cohort_pretrain_medium_1_1_1.sbatch
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=jenkins run_snn_full_cohort_pretrain_medium_1_1_1.sbatch
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=nitschke run_snn_full_cohort_pretrain_medium_1_1_1.sbatch

set -eo pipefail

# --- Activate Environment --
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- Select Which Experiment/Subjects --
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy run_full_cohort_pretrain.sbatch" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy run_full_cohort_pretrain.sbatch" >&2
    exit 1
fi

# --- Customizable Src/Dest Paths ---
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
POOL_DIR="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
CHECKPOINT_DIR="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_pretrained_medium_1_1_1"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
EXP_NAME="bmi_iaf_hard"

# --- Selected Model Architecture and Parameters ---
HIDDEN_DIMS="256 128"
THRESHOLDS="1.0 1.0 1.0"
TAU_SYN=1.0
TEMPORAL_DECAY_STAGES=2
BATCH_SIZE=20
POOLED_EPOCHS=50
POOLED_PATIENCE=5
POOLED_SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=10

if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist -- check EXPERIMENT/SUBJECT and that this "
    echo "       subject's mua_8_group datasets have actually been built." >&2
    exit 1
fi

# --- Build the Pool Dataset ---
if [ -d "$POOL_DIR/train" ] && [ -n "$(ls -A "$POOL_DIR/train" 2>/dev/null)" ]; then
    echo "[reuse] $POOL_DIR already exists and has content, not rebuilding"
else
    if [ -d "$POOL_DIR/train" ]; then
        echo "[rebuild] $POOL_DIR/train exists but is empty -- rebuilding"
    fi
    python3 build_pretraining_pool.py \
        --mua-large-root "$SNN_DATASET_ROOT" \
        --output-dir "$POOL_DIR"
fi

# --- Check if Final Checkpoint Already Exists. If Not, Resume From Last Checkpoint ---
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth"
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] $FINAL_CHECKPOINT already exists (training genuinely completed) -- "
    echo "       delete it (or the whole checkpoint dir) first if you actually want "
    echo "       to retrain this."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] no final checkpoint, but found an incomplete run -- resuming from "
    echo "         periodic checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
elif [ -f "${CHECKPOINT_DIR}/best_model_weights.pth" ]; then
    echo "[resume] no periodic checkpoint found, but best_model_weights.pth exists -- "
    echo "         resuming from that instead."
    RESUME_ARGS+=(--resume "${CHECKPOINT_DIR}/best_model_weights.pth")
else
    echo "[fresh] no final, periodic, or best checkpoint found -- starting fresh"
fi

# --- Start Training ---
echo "Pretraining ${EXPERIMENT}/${SUBJECT}'s full cohort -- pool-dir=$POOL_DIR"
echo "  hidden-dims=$HIDDEN_DIMS thresholds=$THRESHOLDS tau_syn=$TAU_SYN "
echo "  temporal-decay-stages=$TEMPORAL_DECAY_STAGES batch-size=$BATCH_SIZE"
echo "  epochs=$POOLED_EPOCHS patience=$POOLED_PATIENCE checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_bmi_no_tau_syn.py \
    --data-path "$POOL_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn multi \
    --min-vmem -1 \
    --use-iaf-squeeze \
    --weight-init kaiming \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --training-mode windowed \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$POOLED_EPOCHS" \
    --patience "$POOLED_PATIENCE" \
    --scheduler-patience "$POOLED_SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type iaf \
    --tau-syn "$TAU_SYN" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    "${RESUME_ARGS[@]}"
