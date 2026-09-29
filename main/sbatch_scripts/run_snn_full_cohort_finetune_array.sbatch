#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J FULL_COHORT_FINETUNE
#SBATCH -o logs/FULL_COHORT_FINETUNE_%A_%a.out
#SBATCH -e logs/FULL_COHORT_FINETUNE_%A_%a.out
#
# Picks up training from pretrained full cohort checkpoint (only weights initialized).
# Fine-tunes session-specific model for each session in an experiment, using the full
# cohort model as a starting point.
#
# Checkpoints go to checkpoints/bmi/{subject}/full_cohort_finetuned/{session}/
#
# Submit with:
#   mkdir -p logs
#   N_SESSIONS=$(find datasets/bmi/indy/mua_8_group -maxdepth 1 -mindepth 1 -type d | wc -l)
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0 run_snn_full_cohort_finetune_array.sbatch                     # validate one session first
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=1-$((N_SESSIONS-1)) run_snn_full_cohort_finetune_array.sbatch   # the rest

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- Confirm Proper Experiment/Subject to Finetune ---
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_full_cohort_finetune_array.sbatch" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_full_cohort_finetune_array.sbatch" >&2
    exit 1
fi

# --- Customizable Src/Dest Paths ---
SNN_DATASET_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
CHECKPOINTS_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_finetuned"
PRETRAINED_CHECKPOINT="/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/full_cohort_pretrained/best_model_weights.pth"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
EXP_NAME="bmi_iaf_hard"

# --- Selected Model Architecture and Parameters (Must Match Pretrained) ---
HIDDEN_DIMS="256 128 64"
THRESHOLDS="1.0 1.0 1.0 1.0"
TEMPORAL_DECAY_STAGES=2
BATCH_SIZE=20
CHECKPOINT_INTERVAL=10

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_CHECKPOINT -- pretrain "
    echo "       this subject first (HIDDEN_DIMS/THRESHOLDS below must match it)." >&2
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
    echo "[skip] ${SESSION_ID}: $FINAL_CHECKPOINT already exists (fine-tuning genuinely "
    echo "       completed) -- delete it (or the whole checkpoint dir) first if you "
    echo "       actually want to redo this."
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] ${SESSION_ID}: no final checkpoint, but found an incomplete "
    echo "         fine-tuning run -- resuming from $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
else
    echo "[fresh] ${SESSION_ID}: no final or periodic checkpoint found -- starting fresh "
    echo "        from the pretrained model's weights"
fi

# --- Only Run if Starting Train From Scratch ---
INIT_ARGS=()
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    INIT_ARGS+=(--init-weights-from "$PRETRAINED_CHECKPOINT")
fi

# --- Start Training ---
echo "Array task $SLURM_ARRAY_TASK_ID -> experiment=$EXPERIMENT subject=$SUBJECT session=$SESSION_ID ($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS})"
echo "  init-weights-from=$PRETRAINED_CHECKPOINT batch-size=$BATCH_SIZE"
echo "  checkpoint-dir=$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_snn.py \
    --data-path "$SESSION_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn multi \
    --min-vmem -1 \
    --use-iaf-squeeze \
    --weight-init kaiming \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --neuron-type iaf \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    "${INIT_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
