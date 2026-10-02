# Shared SNN configuration for pooled pretraining and per-session fine-tuning.
# Sourced by run_snn_pooled_pretrain.sbatch and run_snn_pooled_finetune_array.sbatch
# (from the submission directory, which Slurm uses as the working directory),
# so both stages always train the same "medium" network with the same settings.
#
# Requires DATA_ROOT, EXPERIMENT and SUBJECT; honours TAU_SYN (default none) and
# RESET_TYPE (hard, the default, or soft: subtract the threshold on a spike).

CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"

# --- Reset after a spike: hard (to zero) or soft (subtract the threshold) ---
RESET_TYPE="${RESET_TYPE:-hard}"
case "$RESET_TYPE" in
    hard) RESET_SUFFIX="" ;;
    soft) RESET_SUFFIX="_soft" ;;
    *) echo "ERROR: RESET_TYPE must be hard or soft, got '$RESET_TYPE'" >&2; exit 1 ;;
esac
EXP_NAME="bmi_iaf_${RESET_TYPE}"   # file prefix train_snn.py uses (bmi_{neuron}_{reset})

# --- Medium topology ---
HIDDEN_DIMS=(256 128)
THRESHOLDS=(1.25 1.25 1.25)   # one per spiking layer: hidden + output
TEMPORAL_DECAY_STAGES=2

# --- Training controls ---
BATCH_SIZE=20
EPOCHS=50
PATIENCE=10
SCHEDULER_PATIENCE=10
CHECKPOINT_INTERVAL=10

# --- tau_syn (initial value; trained per layer) ---
TAU_SYN="${TAU_SYN:-none}"
if [ "$TAU_SYN" = "none" ]; then
    TAU_SYN_SUFFIX=""
    TAU_SYN_ARGS=()
else
    TAU_SYN_SUFFIX="_tau_syn_${TAU_SYN}"
    TAU_SYN_ARGS=(--tau-syn "$TAU_SYN")
fi

SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
POOL_DIR="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
CHECKPOINT_BASE="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}"
PRETRAIN_DIR="${CHECKPOINT_BASE}/full_cohort_pretrained_medium${TAU_SYN_SUFFIX}${RESET_SUFFIX}"
FINETUNE_ROOT="${CHECKPOINT_BASE}/full_cohort_finetuned_medium${TAU_SYN_SUFFIX}${RESET_SUFFIX}"

# Arguments shared by every train_snn.py call in both stages.
TRAIN_ARGS=(
    --config "$CONFIG_YAML"
    --spike-fn multi
    --min-vmem -1
    --use-iaf-squeeze
    --weight-init kaiming
    --neuron-type iaf
    --reset-type "$RESET_TYPE"
    --batch-size "$BATCH_SIZE"
    --num-workers 0
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --scheduler-patience "$SCHEDULER_PATIENCE"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES"
    --hidden-dims "${HIDDEN_DIMS[@]}"
    --spike-thresholds "${THRESHOLDS[@]}"
    "${TAU_SYN_ARGS[@]}"
)

# latest_resume_args DIR -> sets RESUME_ARGS to resume from DIR's newest
# periodic checkpoint, if any.
latest_resume_args() {
    local latest
    RESUME_ARGS=()
    latest=$(ls "$1"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$latest" ]; then
        echo "[resume] $latest"
        RESUME_ARGS=(--resume "$latest")
    fi
}
