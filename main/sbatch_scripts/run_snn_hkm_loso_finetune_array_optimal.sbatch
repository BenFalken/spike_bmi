#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J HKM_LOSO_FINETUNE_OPTIMAL
#SBATCH -o hkm_loso_finetune_optimal_logs/HKM_LOSO_FINETUNE_OPTIMAL_%A_%a.out
#SBATCH -e hkm_loso_finetune_optimal_logs/HKM_LOSO_FINETUNE_OPTIMAL_%A_%a.out
#
# HKM leave-one-session-out fine-tuning, one session per array task, within one subject
# (SUBJECT=jenkins or nitschke). Array task i fine-tunes on session S_i starting from the
# model pretrained on every OTHER session of that subject
# (run_snn_hkm_loso_pretrain_array_optimal.sbatch, same array index).
# Weights only are initialized; optimizer/scheduler start fresh.
#
# Before training, the pretraining pool's pool_manifest.json is checked to
# confirm S_i really was excluded -- the job refuses to run otherwise.
#
# Architecture and parameters match run_snn_hkm_loso_pretrain_array_optimal.sbatch:
#   Hidden layers:  512 -> 256 -> 128
#   Threshold:      1.0 (all layers)
#   Reset:          hard (set in CONFIG_YAML, iaf_hard_reset.yaml)
#   EMA stages:     2
#   tau_syn:        8.0
#
# Pretrained weights are read from
#   snn_checkpoints/<EXPERIMENT>/<SUBJECT>/loso_pretrained_optimal/<SESSION>/best_model_weights.pth
# Checkpoints go to
#   snn_checkpoints/<EXPERIMENT>/<SUBJECT>/loso_finetuned_optimal/<SESSION>/
#
# Submit chained after pretraining (see run_snn_hkm_loso_pretrain_array_optimal.sbatch):
#   sbatch --dependency=aftercorr:$PRE --export=ALL,SUBJECT=jenkins --array=0-3 run_snn_hkm_loso_finetune_array_optimal.sbatch
#   sbatch --export=ALL,SUBJECT=jenkins --array=0-3 run_snn_hkm_loso_finetune_array_optimal.sbatch   # jenkins has 4 sessions
# Set ALLOW_INCOMPLETE_PRETRAIN=1 to fine-tune from a best_model_weights.pth whose
# pretraining run has not written its final_model yet.

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Select Which Experiment/Subject ---
USAGE="sbatch --export=ALL,SUBJECT=jenkins --array=0-N run_snn_hkm_loso_finetune_array_optimal.sbatch"
EXPERIMENT="${EXPERIMENT:-hkm}"
if [ "$EXPERIMENT" != "hkm" ]; then
    echo "ERROR: this script is HKM-only (EXPERIMENT=$EXPERIMENT) -- use run_snn_loso_finetune_array_optimal.sbatch for bmi." >&2
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
# Prefix train_hkm.py actually uses for its output files: f"bmi_{neuron_type}_{reset_type}"
# (printed as "Starting experiment: ..."), i.e. "bmi_iaf_hard" here even though this is HKM.
# This is NOT the same as EXP_NAME -- EXP_NAME only names directories. Must match, or the
# skip/resume/done checks below never find the files train_hkm.py writes.
RUN_TAG="${RUN_TAG:-bmi_iaf_hard}"

# --- Customizable Src/Dest Paths ---
DATA_BASE="/users/bfalkenb/scratch/bfalkenb/data"
SNN_DATASET_ROOT="${DATA_BASE}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"   # whole-trial hkm dataset
POOLS_ROOT="${DATA_BASE}/snn_datasets/${EXPERIMENT}/${SUBJECT}/loso_pools"
PRETRAIN_ROOT="${DATA_BASE}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/loso_pretrained_${EXP_NAME}"
CHECKPOINTS_ROOT="${DATA_BASE}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/loso_finetuned_${EXP_NAME}"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"   # same config as run_snn_hkm_optimal.sbatch
POOL_SCRIPT="${POOL_SCRIPT:-../preprocessing_training/build_pretraining_pool_hkm.py}"

# --- Selected Model Architecture and Parameters (Must Match Pretrained) ---
HIDDEN_DIMS="512 256 128"
THRESHOLDS="1.0 1.0 1.0 1.0"   # one per hidden layer + output layer
TAU_SYN=8.0
TEMPORAL_DECAY_STAGES=2        # EMA staging
BATCH_SIZE=1                   # whole HKM trials differ in length -> cannot be batched
FINETUNE_EPOCHS=10             # as run_snn_hkm_optimal.sbatch
FINETUNE_PATIENCE=5
FINETUNE_SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=1

if [ ! -f "$CONFIG_YAML" ]; then
    echo "ERROR: config $CONFIG_YAML not found." >&2
    exit 1
fi
if [ ! -d "$SNN_DATASET_ROOT" ]; then
    echo "ERROR: $SNN_DATASET_ROOT does not exist -- check EXPERIMENT/SUBJECT and that this" >&2
    echo "       subject's whole-trial hkm datasets have actually been built." >&2
    exit 1
fi
if [ ! -f "$POOL_SCRIPT" ]; then
    echo "ERROR: pool script $POOL_SCRIPT not found (override with POOL_SCRIPT=...)." >&2
    exit 1
fi

# --- Resolve Session ---
# Same canonical ordering as the pretraining array -> index i is the same session in both.
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
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: found zero session directories under $SNN_DATASET_ROOT" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session"
    echo "(only $N_SESSIONS sessions found) -- exiting cleanly."
    exit 0
fi

SESSION_ID="${SESSION_IDS[$SLURM_ARRAY_TASK_ID]}"
SESSION_DIR="${SNN_DATASET_ROOT}/${SESSION_ID}"

if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# --- Locate This Session's Own Leave-One-Out Pretrained Model ---
PRETRAIN_DIR="${PRETRAIN_ROOT}/${SESSION_ID}"
PRETRAINED_CHECKPOINT="${PRETRAIN_DIR}/best_model_weights.pth"
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: LOSO pretrained checkpoint not found at $PRETRAINED_CHECKPOINT -- run" >&2
    echo "       run_snn_hkm_loso_pretrain_array_optimal.sbatch (array task $SLURM_ARRAY_TASK_ID) first." >&2
    exit 1
fi
if [ ! -f "${PRETRAIN_DIR}/final_model_${RUN_TAG}.pth" ]; then
    if [ "${ALLOW_INCOMPLETE_PRETRAIN:-0}" != "1" ]; then
        echo "ERROR: ${PRETRAIN_DIR}/final_model_${RUN_TAG}.pth not found -- LOSO pretraining" >&2
        echo "       for ${SESSION_ID} is still running or incomplete. Set ALLOW_INCOMPLETE_PRETRAIN=1" >&2
        echo "       to fine-tune from its current best_model_weights.pth anyway." >&2
        exit 1
    fi
    echo "WARNING: pretraining for ${SESSION_ID} incomplete; using current best_model_weights.pth" >&2
fi

# --- Leakage Guard: Confirm This Session Was Excluded From Its Pretraining Pool ---
MANIFEST="${POOLS_ROOT}/${SESSION_ID}/pool_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found -- cannot verify ${SESSION_ID} was held out of pretraining." >&2
    exit 1
fi
python3 - "$MANIFEST" "$SESSION_ID" <<'PYEOF'
import json, sys
manifest_path, session_id = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
if session_id not in m["excluded_sessions"] or session_id in m["included_sessions"]:
    sys.exit(f"LEAKAGE: {session_id} was NOT excluded from its pretraining pool ({manifest_path})")
print(f"[ok] {session_id} excluded from pretraining pool "
      f"({len(m['included_sessions'])} other sessions pooled)")
PYEOF

# --- Check if Final Checkpoint Already Exists. If Not, Resume From Last Checkpoint ---
CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/${SESSION_ID}"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${RUN_TAG}.pth"
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] ${SESSION_ID}: $FINAL_CHECKPOINT already exists (fine-tuning genuinely"
    echo "       completed) -- delete it (or the whole checkpoint dir) first if you"
    echo "       actually want to redo this."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${RUN_TAG}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
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
