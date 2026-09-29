#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_POOLED_PRETRAIN
#SBATCH -o snn_logs/SNN_POOLED_PRETRAIN_%j.out
#SBATCH -e snn_logs/SNN_POOLED_PRETRAIN_%j.out
#
# Stage 1 of 2: train one SNN on the pooled grouped datasets (mua_8_group)
# of every session of one subject. The network and training settings are the
# "medium" configuration in snn_medium_config.sh (hidden 256 -> 128, spike
# threshold 1.25), shared with run_snn_pooled_finetune_array.sbatch.
#
# The pool is built once with build_pretraining_pool.py (symlinks, no
# copies). Train trials all share one length and are batched; each
# session's single test trial has its own length, so the test set is
# evaluated one trial at a time.
#
# Submit (from this directory):
#   mkdir -p snn_logs
#   sbatch --export=ALL,EXPERIMENT=bmi,SUBJECT=indy run_snn_pooled_pretrain.sbatch
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
source ./snn_medium_config.sh

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
    python3 ../preprocessing_training/build_pretraining_pool.py \
        --mua-large-root "$SNN_DATASET_ROOT" --output-dir "$POOL_DIR"
fi

# --- Skip / resume ---
if [ -f "${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth" ]; then
    echo "[skip] ${PRETRAIN_DIR}/final_model_${EXP_NAME}.pth exists; delete it to retrain."
    exit 0
fi
latest_resume_args "$PRETRAIN_DIR"
if [ ${#RESUME_ARGS[@]} -eq 0 ] && [ -f "${PRETRAIN_DIR}/best_model_weights.pth" ]; then
    echo "[resume] ${PRETRAIN_DIR}/best_model_weights.pth (no periodic checkpoint)"
    RESUME_ARGS=(--resume "${PRETRAIN_DIR}/best_model_weights.pth")
fi

echo "Pooled pretraining ${EXPERIMENT}/${SUBJECT}: hidden=${HIDDEN_DIMS[*]} thresholds=${THRESHOLDS[*]}" \
     "tau_syn=${TAU_SYN} -> ${PRETRAIN_DIR}"
mkdir -p "$PRETRAIN_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$POOL_DIR" \
    --checkpoint-dir "$PRETRAIN_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
