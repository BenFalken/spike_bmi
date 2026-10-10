# The two SNN configurations of the final pipeline, shared by every training
# script (sourced from the submission directory, which Slurm uses as the
# working directory):
#
#   pooled       run_snn_pooled_pretrain.sbatch + run_snn_pooled_finetune_array.sbatch:
#                the PyTorch SNN ('snn' in inference). 512 -> 256 -> 128, threshold
#                1.0, 2 EMA stages, tau_syn initialized at 2, 4, 8, 16 over the four
#                spiking layers (trained). Not deployable to Speck (no synaptic stage).
#   per_session  run_snn_per_session_array.sbatch: the Speck SNN ('speck'). 256 -> 128
#                -> 64, threshold 1.0, 2 EMA stages, no tau_syn.
#
# Every model trains on binarized input (train_snn.py --binarize-input, default).
#
# Requires DATA_ROOT, EXPERIMENT and SUBJECT; call select_model pooled|per_session.
# LOSO (pooled only): 1 (default) pretrains one model per session on every OTHER
# session of the subject, so a session's own data never reaches its pretraining;
# 0 pretrains one model on every session.

CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
POOL_SCRIPT="../preprocessing_training/build_pretraining_pool.py"
EXP_NAME="bmi_iaf_hard"   # file prefix train_snn.py uses (bmi_{neuron}_{reset})

# bmi: grouped datasets, batches of 20 equal-length trials.
# hkm: whole-trial datasets of different lengths, one trial per batch.
case "$EXPERIMENT" in
    hkm) DATASET_SUBDIR="mua"; BATCH_SIZE=1 ;;
    *)   DATASET_SUBDIR="mua_8_group"; BATCH_SIZE=20 ;;
esac
SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/${DATASET_SUBDIR}"
CHECKPOINT_BASE="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}"

LOSO="${LOSO:-1}"
case "$LOSO" in
    0) POOL_KIND="full_cohort" ;;
    1) POOL_KIND="loso" ;;
    *) echo "ERROR: LOSO must be 0 or 1, got '$LOSO'" >&2; exit 1 ;;
esac
# Checkpoint directories end in _binarized (binarized input), so they never
# meet the checkpoints of earlier runs, which the skip/resume checks would
# otherwise take for this pipeline's.
FINETUNE_ROOT="${CHECKPOINT_BASE}/${POOL_KIND}_finetuned_binarized"
PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_binarized"

# select_model pooled|per_session -> the network and training controls.
# Epochs are maxima (early stopping); hkm epochs take about an hour per
# session, so hkm runs fewer and checkpoints every epoch.
select_model() {
    case "$1" in
        pooled)
            HIDDEN_DIMS=(512 256 128); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=(2 4 8 16)
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=10; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=10; CHECKPOINT_INTERVAL=10
            fi ;;
        per_session)
            HIDDEN_DIMS=(256 128 64); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=()
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=20; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10
            fi ;;
        *) echo "ERROR: select_model pooled|per_session, got '$1'" >&2; exit 1 ;;
    esac
    TAU_SYN_ARGS=()
    [ ${#TAU_SYN[@]} -gt 0 ] && TAU_SYN_ARGS=(--tau-syn "${TAU_SYN[@]}")
    TRAIN_ARGS=(
        --config "$CONFIG_YAML"
        --spike-fn multi
        --min-vmem -1
        --use-iaf-squeeze
        --weight-init kaiming
        --neuron-type iaf
        --reset-type hard
        --binarize-input true
        --batch-size "$BATCH_SIZE"
        --num-workers 0
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --scheduler-patience "$SCHEDULER_PATIENCE"
        --checkpoint-interval "$CHECKPOINT_INTERVAL"
        --temporal-decay-stages 2
        --hidden-dims "${HIDDEN_DIMS[@]}"
        --spike-thresholds "${THRESHOLDS[@]}"
        "${TAU_SYN_ARGS[@]}"
    )
    MODEL_DESC="$1: hidden=${HIDDEN_DIMS[*]} thresholds=${THRESHOLDS[*]} tau_syn=${TAU_SYN[*]:-none} batch=${BATCH_SIZE} epochs<=${EPOCHS}"
}

# Pretraining pool and checkpoint directory; per held-out session with LOSO=1.
# Pool directory names contain "pool", which compute_velocity_scalers.py skips.
pool_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/loso_pools/$1"
    else
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
    fi
}
pretrain_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${CHECKPOINT_BASE}/loso_pretrained_binarized/$1"
    else
        echo "${CHECKPOINT_BASE}/full_cohort_pretrained_binarized"
    fi
}

# session_dirs -> SESSION_DIRS: every session of the subject, sorted (array
# task i is session i in every script).
session_dirs() {
    mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    if [ "${#SESSION_DIRS[@]}" -eq 0 ]; then
        echo "ERROR: no session directories under $SNN_DATASET_ROOT" >&2
        exit 1
    fi
}

# array_session -> SESSION_DIR, SESSION_ID for this array task (exits cleanly
# past the last session).
array_session() {
    session_dirs
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: submit as an array, one task per session: --array=0-$((${#SESSION_DIRS[@]} - 1))" >&2
        exit 1
    fi
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SESSION_DIRS[@]}" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#SESSION_DIRS[@]} found); exiting."
        exit 0
    fi
    SESSION_DIR="${SESSION_DIRS[$SLURM_ARRAY_TASK_ID]}"
    SESSION_ID="$(basename "$SESSION_DIR")"
}

# latest_resume_args DIR -> RESUME_ARGS resuming from DIR's newest periodic
# checkpoint, else from its best_model_weights.pth, if any.
latest_resume_args() {
    local latest
    RESUME_ARGS=()
    latest=$(ls "$1"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    [ -z "$latest" ] && [ -f "$1/best_model_weights.pth" ] && latest="$1/best_model_weights.pth"
    if [ -n "$latest" ]; then
        echo "[resume] $latest"
        RESUME_ARGS=(--resume "$latest")
    fi
}

activate_env() {
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate snn_speck
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
}
