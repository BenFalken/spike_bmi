#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J HKM_LOSO_PRETRAIN_OPTIMAL
#SBATCH -o hkm_loso_pretrain_optimal_logs/HKM_LOSO_PRETRAIN_OPTIMAL_%A_%a.out
#SBATCH -e hkm_loso_pretrain_optimal_logs/HKM_LOSO_PRETRAIN_OPTIMAL_%A_%a.out
#
# HKM leave-one-session-out pretraining, one held-out session per array task, WITHIN one
# subject (pass SUBJECT=jenkins or SUBJECT=nitschke; pools never mix subjects).
# Mirrors run_snn_loso_pretrain_array_optimal.sbatch (BMI) with HKM's trainer/data/settings:
# whole-trial dataset (.../hkm/<subject>/mua), train_hkm.py, batch size 1 (trials differ in
# length, so they cannot be batched), and the HKM optimal config of run_snn_hkm_optimal.sbatch.
# Array task i:
#   1. resolves held-out session S_i (canonical order from build_pretraining_pool_hkm.py --list-sessions)
#   2. builds a symlink pool of every session EXCEPT S_i
#   3. pretrains from scratch on that pool
# S_i's data is never read here, so it can't influence pretraining's gradients
# OR its best_loss checkpoint selection.
#
# Architecture and parameters match the "optimal" config:
#   Hidden layers:  512 -> 256 -> 128
#   Threshold:      1.0 (all layers)
#   Reset:          hard (set in CONFIG_YAML, iaf_hard_reset.yaml)
#   EMA stages:     2
#   tau_syn:        8.0
#
# Pools go to
#   snn_datasets/<EXPERIMENT>/<SUBJECT>/loso_pools/<SESSION>/
# Checkpoints go to
#   snn_checkpoints/<EXPERIMENT>/<SUBJECT>/loso_pretrained_optimal/<SESSION>/
#
# Submit (pretrain + fine-tune chained per task with aftercorr):
#   mkdir -p hkm_loso_pretrain_optimal_logs hkm_loso_finetune_optimal_logs
#   ROOT=/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/hkm/jenkins/mua
#   N_SESSIONS=$(python ../preprocessing_training/build_pretraining_pool_hkm.py --mua-large-root "$ROOT" --list-sessions | wc -l)
#   PRE=$(sbatch --parsable --export=ALL,SUBJECT=jenkins --array=0-$((N_SESSIONS-1)) run_snn_hkm_loso_pretrain_array_optimal.sbatch)
#   sbatch --dependency=aftercorr:$PRE --export=ALL,SUBJECT=jenkins --array=0-$((N_SESSIONS-1)) run_snn_hkm_loso_finetune_array_optimal.sbatch
# (aftercorr: fine-tune task i starts only once pretrain task i finishes successfully.)
#
# RUNTIME: batch size 1 over a pool of every other session means one epoch takes hours
# (a single HKM session was ~50 min/epoch). Checkpoints are written EVERY epoch
# (CHECKPOINT_INTERVAL=1) and this script resumes from the newest one, so if the 24 h limit
# hits, just resubmit the same array: finished tasks skip, unfinished ones continue.

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Select Which Experiment/Subject ---
USAGE="sbatch --export=ALL,SUBJECT=jenkins --array=0-N run_snn_hkm_loso_pretrain_array_optimal.sbatch"
EXPERIMENT="${EXPERIMENT:-hkm}"
if [ "$EXPERIMENT" != "hkm" ]; then
    echo "ERROR: this script is HKM-only (EXPERIMENT=$EXPERIMENT) -- use run_snn_loso_pretrain_array_optimal.sbatch for bmi." >&2
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

# --- Configuration Name ---
EXP_NAME="optimal"
# Prefix train_hkm.py actually uses for its output files: f"bmi_{neuron_type}_{reset_type}"
# (printed as "Starting experiment: ..."), i.e. "bmi_iaf_hard" here even though this is HKM.
# This is NOT the same as EXP_NAME -- EXP_NAME only names directories. Must match, or the
# skip/resume/done checks below never find the files train_hkm.py writes.
RUN_TAG="${RUN_TAG:-bmi_iaf_hard}"

# --- Customizable Src/Dest Paths ---
DATA_BASE="/users/bfalkenb/scratch/bfalkenb/data"
SNN_DATASET_ROOT="${DATA_BASE}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"   # whole-trial hkm dataset
POOLS_ROOT="${DATA_BASE}/snn_datasets/${EXPERIMENT}/${SUBJECT}/loso_pools"
CHECKPOINTS_ROOT="${DATA_BASE}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/loso_pretrained_${EXP_NAME}"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"   # same config as run_snn_hkm_optimal.sbatch
POOL_SCRIPT="${POOL_SCRIPT:-../preprocessing_training/build_pretraining_pool_hkm.py}"

# --- Selected Model Architecture and Parameters ---
HIDDEN_DIMS="512 256 128"
THRESHOLDS="1.0 1.0 1.0 1.0"   # one per hidden layer + output layer
TAU_SYN=8.0
TEMPORAL_DECAY_STAGES=2        # EMA staging
BATCH_SIZE=1                   # whole HKM trials differ in length -> cannot be batched
PRETRAIN_EPOCHS=10             # as run_snn_hkm_optimal.sbatch; an epoch over the pool is long
PRETRAIN_PATIENCE=5
PRETRAIN_SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=1          # every epoch, so a 24 h timeout loses at most one epoch

if [ ! -f "$CONFIG_YAML" ]; then
    echo "ERROR: config $CONFIG_YAML not found." >&2
    exit 1
fi
if [ ! -f "$POOL_SCRIPT" ]; then
    echo "ERROR: pool script $POOL_SCRIPT not found (override with POOL_SCRIPT=...)." >&2
    exit 1
fi
if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist -- check EXPERIMENT/SUBJECT and that this" >&2
    echo "       subject's whole-trial hkm datasets have actually been built." >&2
    exit 1
fi

# --- Resolve Held-Out Session (same ordering as the pool script and fine-tune array) ---
# Run the listing as a normal command (not inside <(...)) so a failure is caught
# rather than silently producing an empty list.
if ! grep -q -- "--list-sessions" "$POOL_SCRIPT"; then
    echo "ERROR: $POOL_SCRIPT is an outdated build_pretraining_pool_hkm.py (no --list-sessions" >&2
    echo "       option) -- replace it with the updated LOSO version." >&2
    exit 1
fi
if ! SESSION_LIST=$(python3 "$POOL_SCRIPT" --mua-large-root "$SNN_DATASET_ROOT" --list-sessions); then
    echo "ERROR: '$POOL_SCRIPT --list-sessions' failed (see its error above)." >&2
    exit 1
fi
mapfile -t SESSION_IDS <<< "$SESSION_LIST"
[ -z "$SESSION_LIST" ] && SESSION_IDS=()
N_SESSIONS=${#SESSION_IDS[@]}
if [ "$N_SESSIONS" -lt 2 ]; then
    echo "ERROR: found $N_SESSIONS session(s) under $SNN_DATASET_ROOT -- leave-one-out needs at least 2." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session"
    echo "(only $N_SESSIONS sessions found) -- exiting cleanly."
    exit 0
fi

HELDOUT_ID="${SESSION_IDS[$SLURM_ARRAY_TASK_ID]}"
POOL_DIR="${POOLS_ROOT}/${HELDOUT_ID}"
CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/${HELDOUT_ID}"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${RUN_TAG}.pth"

# --- Skip if Already Done ---
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] held-out=${HELDOUT_ID}: $FINAL_CHECKPOINT already exists -- delete it"
    echo "       (or the whole checkpoint dir) first if you actually want to redo this."
    exit 0
fi

# --- Build the Leave-One-Out Pool (rebuilt every start; symlinks only, cheap) ---
echo "Building LOSO pool excluding ${HELDOUT_ID} -> ${POOL_DIR}"
mkdir -p "$POOLS_ROOT"
python3 "$POOL_SCRIPT" \
    --mua-large-root "$SNN_DATASET_ROOT" \
    --output-dir "$POOL_DIR" \
    --exclude-session "$HELDOUT_ID"

# --- Resume From Last Checkpoint if Present ---
RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${RUN_TAG}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] held-out=${HELDOUT_ID}: resuming from periodic checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
elif [ -f "${CHECKPOINT_DIR}/best_model_weights.pth" ]; then
    echo "[resume] held-out=${HELDOUT_ID}: no periodic checkpoint, resuming from best_model_weights.pth"
    RESUME_ARGS+=(--resume "${CHECKPOINT_DIR}/best_model_weights.pth")
else
    echo "[fresh] held-out=${HELDOUT_ID}: no checkpoint found -- pretraining from scratch"
fi

# --- Start Pretraining ---
echo "Array task $SLURM_ARRAY_TASK_ID -> ${EXPERIMENT}/${SUBJECT} held-out=$HELDOUT_ID ($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS}) (config=$EXP_NAME)"
echo "  pool=$POOL_DIR ($((N_SESSIONS-1)) sessions)"
echo "  hidden-dims=[$HIDDEN_DIMS] thresholds=[$THRESHOLDS] tau-syn=$TAU_SYN"
echo "  ema-stages=$TEMPORAL_DECAY_STAGES batch-size=$BATCH_SIZE"
echo "  epochs=$PRETRAIN_EPOCHS patience=$PRETRAIN_PATIENCE checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_hkm.py \
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
    --epochs "$PRETRAIN_EPOCHS" \
    --patience "$PRETRAIN_PATIENCE" \
    --scheduler-patience "$PRETRAIN_SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type iaf \
    --tau-syn "$TAU_SYN" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    "${RESUME_ARGS[@]}"
