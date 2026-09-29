#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_PRETRAIN
#SBATCH -o snn_logs/SNN_POOLED_PRETRAIN_%j.out
#SBATCH -e snn_logs/SNN_POOLED_PRETRAIN_%j.out
#
# Train one SNN on the pooled grouped datasets (mua_8_group) of every
# session of one subject, using the "medium" configuration:
# hidden 256 -> 128, spike threshold 1.25, 2 readout decay stages.
#
# The pool is built once with build_pretraining_pool.py (symlinks, no
# copies). Train trials all share one length and are batched; each
# session's single test trial has its own length, so the test set is
# evaluated one trial at a time.
#
# Submit (from this directory):
#   mkdir -p snn_logs
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy run_snn_pooled_pretrain.sbatch
#   sbatch --export=ALL,EXPERIMENT=hkm,SUBJECT=jenkins run_snn_pooled_pretrain.sbatch
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy,TAU_SYN=8.0 run_snn_pooled_pretrain.sbatch
#
# Checkpoints: ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/full_cohort_pretrained_medium[_tau_syn_<T>]/
# Re-submitting skips a finished run and resumes an interrupted one.

set -eo pipefail

if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "ERROR: set EXPERIMENT and SUBJECT, e.g." >&2
    echo "  sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy $0" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
POOL_DIR="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
POOL_SCRIPT="../preprocessing_training/build_pretraining_pool.py"
EXP_NAME="bmi_iaf_hard"   # file prefix train_snn.py uses for an IAF / hard-reset model

# --- Medium configuration ---
HIDDEN_DIMS=(256 128)
THRESHOLDS=(1.25 1.25 1.25)   # one per spiking layer: hidden + output
TEMPORAL_DECAY_STAGES=2
BATCH_SIZE=20
EPOCHS=50
PATIENCE=10
SCHEDULER_PATIENCE=10
CHECKPOINT_INTERVAL=10

TAU_SYN="${TAU_SYN:-none}"
CHECKPOINT_DIR="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_pretrained_medium"
TAU_SYN_ARGS=()
if [ "$TAU_SYN" != "none" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR}_tau_syn_${TAU_SYN}"
    TAU_SYN_ARGS=(--tau-syn "$TAU_SYN")
fi

if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist; build the grouped datasets first." >&2
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Pool (built once) ---
if [ -n "$(ls -A "$POOL_DIR/train" 2>/dev/null)" ]; then
    echo "[reuse] pool at $POOL_DIR"
else
    python3 "$POOL_SCRIPT" --mua-large-root "$SNN_DATASET_ROOT" --output-dir "$POOL_DIR"
fi

# --- Skip / resume ---
if [ -f "${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth exists; delete it to retrain."
    exit 0
fi
RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] $LATEST_CHECKPOINT"
    RESUME_ARGS=(--resume "$LATEST_CHECKPOINT")
elif [ -f "${CHECKPOINT_DIR}/best_model_weights.pth" ]; then
    echo "[resume] ${CHECKPOINT_DIR}/best_model_weights.pth (no periodic checkpoint)"
    RESUME_ARGS=(--resume "${CHECKPOINT_DIR}/best_model_weights.pth")
else
    echo "[fresh] no checkpoint found"
fi

echo "Pooled pretraining ${EXPERIMENT}/${SUBJECT}: hidden=${HIDDEN_DIMS[*]} thresholds=${THRESHOLDS[*]}" \
     "tau_syn=${TAU_SYN} decay_stages=${TEMPORAL_DECAY_STAGES} -> ${CHECKPOINT_DIR}"
mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$POOL_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn multi \
    --min-vmem -1 \
    --use-iaf-squeeze \
    --weight-init kaiming \
    --neuron-type iaf \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --scheduler-patience "$SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims "${HIDDEN_DIMS[@]}" \
    --spike-thresholds "${THRESHOLDS[@]}" \
    "${TAU_SYN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
