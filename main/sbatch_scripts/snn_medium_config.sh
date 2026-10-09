# Shared SNN configuration for pooled pretraining and per-session fine-tuning.
# Sourced by run_snn_pooled_pretrain.sbatch and run_snn_pooled_finetune_array.sbatch
# (from the submission directory, which Slurm uses as the working directory),
# so both stages always train the same "medium" network with the same settings.
#
# Requires DATA_ROOT, EXPERIMENT and SUBJECT; honours TAU_SYN (default none),
# RESET_TYPE (hard, the default, or soft: subtract the threshold on a spike)
# and LOSO (0, the default: one model pretrained on every session; 1: one
# model per session, pretrained on every OTHER session of the subject, so a
# session's own data never reaches its pretraining, not even through the
# choice of the best checkpoint).

CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
POOL_SCRIPT="../preprocessing_training/build_pretraining_pool.py"

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

# --- Datasets and training controls ---
# bmi: grouped datasets, batches of 20 equal-length trials.
# hkm: whole-trial datasets (nwb_conversion/), one trial per batch since
#      trials differ in length. An epoch over a pool takes hours, so hkm runs
#      fewer epochs and checkpoints every epoch; re-submitting resumes.
case "$EXPERIMENT" in
    hkm)
        DATASET_SUBDIR="mua"
        BATCH_SIZE=1; EPOCHS=10; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1 ;;
    *)
        DATASET_SUBDIR="mua_8_group"
        BATCH_SIZE=20; EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=10; CHECKPOINT_INTERVAL=10 ;;
esac

# --- tau_syn (initial value; trained per layer) ---
TAU_SYN="${TAU_SYN:-none}"
if [ "$TAU_SYN" = "none" ]; then
    TAU_SYN_SUFFIX=""
    TAU_SYN_ARGS=()
else
    TAU_SYN_SUFFIX="_tau_syn_${TAU_SYN}"
    TAU_SYN_ARGS=(--tau-syn "$TAU_SYN")
fi

# --- Pools and checkpoints ---
LOSO="${LOSO:-0}"
case "$LOSO" in
    0) POOL_KIND="full_cohort" ;;
    1) POOL_KIND="loso" ;;
    *) echo "ERROR: LOSO must be 0 or 1, got '$LOSO'" >&2; exit 1 ;;
esac
SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/${DATASET_SUBDIR}"
CHECKPOINT_BASE="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}"
VARIANT="medium${TAU_SYN_SUFFIX}${RESET_SUFFIX}"
FINETUNE_ROOT="${CHECKPOINT_BASE}/${POOL_KIND}_finetuned_${VARIANT}"

# Pretraining pool and checkpoint directory. With LOSO=1 they are per held-out
# session: pool_dir SESSION, pretrain_dir SESSION. Pool directory names
# contain "pool", which compute_velocity_scalers.py skips.
pool_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/loso_pools/$1"
    else
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
    fi
}
pretrain_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${CHECKPOINT_BASE}/loso_pretrained_${VARIANT}/$1"
    else
        echo "${CHECKPOINT_BASE}/full_cohort_pretrained_${VARIANT}"
    fi
}

# session_dirs -> SESSION_DIRS: every session of the subject, sorted (array
# task i is session i in both stages).
session_dirs() {
    mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    if [ "${#SESSION_DIRS[@]}" -eq 0 ]; then
        echo "ERROR: no session directories under $SNN_DATASET_ROOT" >&2
        exit 1
    fi
}

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
