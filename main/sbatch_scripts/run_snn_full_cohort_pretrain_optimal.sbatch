#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J FULL_COHORT_PRETRAIN_OPTIMAL
#SBATCH -o full_pretrain_optimal_logs/FULL_COHORT_PRETRAIN_OPTIMAL_%j.out
#SBATCH -e full_pretrain_optimal_logs/FULL_COHORT_PRETRAIN_OPTIMAL_%j.out
#
# Pools all sessions of an experiment/subject together and pretrains the
# "optimal" SNN configuration on the full dataset.
#
#   Hidden layers:  512 -> 256 -> 128
#   Threshold:      1.0 (all layers)
#   Reset:          soft (set in CONFIG_YAML)
#   EMA stages:     2
#   tau_syn:        8.0
#
# Merges train and test data in different configurations:
#   - TRAIN: every session's own train trials are already the same
#     length (group_size=8). Pooling introduces no length
#     mismatch, enabling training in parallel (batches of size >1).
#   - TEST: each session's test split is its own single, whole,
#     unchunked trial. Sessions have different recording durations,
#     so a pooled test set has genuinely different
#     trial lengths across sessions. Batch size is forced to 1 in
#     instances where trial length differs.
#
# Checkpoints go to snn_checkpoints/<EXPERIMENT>/<SUBJECT>/full_cohort_pretrained_optimal/
#
# Submit with (full_pretrain_optimal_logs/ must exist before submission):
#   mkdir -p full_pretrain_optimal_logs
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy run_snn_full_cohort_pretrain_optimal.sbatch
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=loco run_snn_full_cohort_pretrain_optimal.sbatch
#   sbatch --export=ALL,EXPERIMENT=hkm,SUBJECT=jenkins run_snn_full_cohort_pretrain_optimal.sbatch
#   sbatch --export=ALL,EXPERIMENT=hkm,SUBJECT=nitschke run_snn_full_cohort_pretrain_optimal.sbatch

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Select Which Experiment/Subject ---
USAGE="sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy run_snn_full_cohort_pretrain_optimal.sbatch"
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

# --- Configuration Name ---
EXP_NAME="optimal"

# --- Customizable Src/Dest Paths ---
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
POOL_DIR="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
CHECKPOINT_DIR="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_pretrained_${EXP_NAME}"
CONFIG_YAML="../configs/iaf_soft_reset.yaml"   # soft-reset IAF config whose experiment name is "optimal"

# --- Selected Model Architecture and Parameters ---
HIDDEN_DIMS="512 256 128"
THRESHOLDS="1.0 1.0 1.0 1.0"   # one per hidden layer + output layer
TAU_SYN=8.0
TEMPORAL_DECAY_STAGES=2        # EMA staging
BATCH_SIZE=20
POOLED_EPOCHS=50
POOLED_PATIENCE=10
POOLED_SCHEDULER_PATIENCE=10
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
    echo "[skip] $FINAL_CHECKPOINT already exists (training genuinely completed) --"
    echo "       delete it (or the whole checkpoint dir) first if you actually want"
    echo "       to retrain this."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] no final checkpoint, but found an incomplete run -- resuming from"
    echo "         periodic checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
elif [ -f "${CHECKPOINT_DIR}/best_model_weights.pth" ]; then
    echo "[resume] no periodic checkpoint found, but best_model_weights.pth exists --"
    echo "         resuming from that instead."
    RESUME_ARGS+=(--resume "${CHECKPOINT_DIR}/best_model_weights.pth")
else
    echo "[fresh] no final, periodic, or best checkpoint found -- starting fresh"
fi

# --- Start Training ---
echo "Pretraining ${EXPERIMENT}/${SUBJECT}'s full cohort (config=$EXP_NAME) -- pool-dir=$POOL_DIR"
echo "  hidden-dims=[$HIDDEN_DIMS] thresholds=[$THRESHOLDS] tau-syn=$TAU_SYN"
echo "  ema-stages=$TEMPORAL_DECAY_STAGES batch-size=$BATCH_SIZE"
echo "  epochs=$POOLED_EPOCHS patience=$POOLED_PATIENCE checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_bmi.py \
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
