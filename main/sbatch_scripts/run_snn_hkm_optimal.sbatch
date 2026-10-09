#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_HKM_NO_TAU_SYN
#SBATCH -o snn_hkm_optimal_logs/SNN_HKM_NO_TAU_SYN_%A_%a.out
#SBATCH -e snn_hkm_optimal_logs/SNN_HKM_NO_TAU_SYN_%A_%a.out
#
# Trains SNN session-by-session with same parameters as pretrained model.
#
# Submit with:
#   mkdir -p snn_hkm_optimal_logs
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=jenkins --array=0-6 run_snn_hkm_optimal.sbatch
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=nitschke --array=0-6 run_snn_hkm_optimal.sbatch

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Resolve Experiment/Subject ---
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_snn_hkm.sbatch" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_snn_hkm.sbatch" >&2
    exit 1
fi

# --- Customizable Src/Dest Paths ---
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"
CHECKPOINTS_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/optimal"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
# Prefix train_hkm.py actually uses for its output files ("Starting experiment: bmi_iaf_hard"
# in the log) -- NOT "hkm_iaf_hard". Must match, or the skip/resume checks below never find
# the files the trainer writes.
EXP_NAME="${EXP_NAME:-bmi_iaf_hard}"

# --- Selected Model Architecture and Parameters ---
HIDDEN_DIMS="512 256 128"
THRESHOLDS="1.0 1.0 1.0 1.0"   # one per hidden layer + output layer
TAU_SYN=8.0
TEMPORAL_DECAY_STAGES=2        # EMA staging
BATCH_SIZE=1
EPOCHS=20
PATIENCE=5
SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=5   # an epoch takes ~1h+ at batch size 1; save every epoch so a timeout loses little

if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist -- check EXPERIMENT/SUBJECT and that "
    echo "       this subject's mua_8_group datasets have actually been built." >&2
    exit 1
fi

# --- Resolve Session ---
mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)
N_SESSIONS=${#SESSION_DIRS[@]}
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: found zero session directories under $SNN_DATASET_ROOT" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session "
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
    echo "[skip] ${SESSION_ID}: $FINAL_CHECKPOINT already exists -- delete it "
    echo "       (or the whole checkpoint dir) first if you actually want to retrain."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
BEST_CHECKPOINT="${CHECKPOINT_DIR}/best_model_weights.pth"
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] ${SESSION_ID}: resuming from periodic checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
elif [ -f "$BEST_CHECKPOINT" ]; then
    # Full checkpoint (model + optimizer + epoch + losses), written whenever test loss improves.
    echo "[resume] ${SESSION_ID}: no periodic checkpoint, resuming from $BEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$BEST_CHECKPOINT")
else
    echo "[fresh] ${SESSION_ID}: no checkpoint found -- starting fresh"
fi

# --- Start Training ---
echo "Array task $SLURM_ARRAY_TASK_ID -> experiment=$EXPERIMENT subject=$SUBJECT session=$SESSION_ID ($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS})"
echo "  batch-size=$BATCH_SIZE checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_hkm.py \
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
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --scheduler-patience "$SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type iaf \
    --tau-syn "$TAU_SYN" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    "${RESUME_ARGS[@]}"
