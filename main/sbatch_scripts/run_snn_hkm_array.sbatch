#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_HKM
#SBATCH -o snn_logs/SNN_HKM_%A_%a.out
#SBATCH -e snn_logs/SNN_HKM_%A_%a.out
#
# Trains a series of SNN models (HKM_TABLE) on every session of every HKM
# subject, one array task per (session, model).
#
# HKM SNN datasets are whole, variable-length trials, one .pkl per NWB trial
# (nwb_conversion/make_snn_dataset_whole_trial.py):
#   ${DATA_ROOT}/snn_datasets/hkm/<subject>/mua/<session>/{train,test}/*.pkl
# Trials of different lengths cannot be stacked, so every model trains with
# --batch-size 1: one trial per optimizer step, the model state reset at the
# start of each trial. models/model_hkm.py is selected from the "hkm" in the
# dataset path.
#
# Prerequisite: velocity_scalers.json must hold an hkm/<subject> entry for
# each subject (plan mode checks this), e.g. from main/preprocessing_training:
#   python compute_velocity_scalers.py --snn-datasets-root $DATA_ROOT/snn_datasets \
#       --experiments bmi hkm --output ../snn_training/velocity_scalers.json
# (name bmi too: the output file is replaced, not merged).
#
# Two-step workflow, from this directory
# --------------------------------------
# 1. PLAN (plain bash; submits nothing):
#      bash run_snn_hkm_array.sbatch --plan              # every hkm subject
#      bash run_snn_hkm_array.sbatch --plan jenkins      # one subject
#      HKM_FILTER='^medium' bash run_snn_hkm_array.sbatch --plan
#    Lists every (session, model) pair without a final checkpoint in a
#    manifest under snn_logs/manifests/ and prints the sbatch command.
#    HKM_FILTER is an optional bash regex on model names.
# 2. SUBMIT the printed command, e.g.
#      sbatch --export=ALL,MANIFEST=/abs/path/manifest.txt --array=0-47 run_snn_hkm_array.sbatch
#    To validate first, submit --array=0 alone, then the rest.
#
# Each task skips a finished model and resumes an interrupted one from its
# latest epoch checkpoint, so after timeouts re-plan and re-submit.
#
# Outputs, one directory per (session, model):
#   ${DATA_ROOT}/snn_checkpoints/hkm/<subject>/hkm_series/<session>/<model>/
#       best_model_weights.pth, final_model_bmi_iaf_<reset>.pth, checkpoint_*_epoch<N>.pth
# Evaluate one model with run_inference.sbatch:
#   SNN_CHECKPOINT_ROOT=$DATA_ROOT/snn_checkpoints/hkm/<subject>/hkm_series \
#   bash run_inference.sbatch --submit hkm <subject> SNN_CHECKPOINT_SUBDIR=<model>

set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
EXPERIMENT="hkm"
DATASET_SUBDIR="mua"            # whole-trial datasets (not mua_8_group)
SERIES_TAG="hkm_series"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
VELOCITY_SCALERS="${BMI_VELOCITY_SCALERS_PATH:-../snn_training/velocity_scalers.json}"

# ---------------------------------------------------------------------------
# Training controls (shared by every model)
# ---------------------------------------------------------------------------
BATCH_SIZE=1                    # required: trials have different lengths
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.001}"
PATIENCE=10
SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=5           # one epoch is one optimizer step per trial, so save often
LOG_INTERVAL=500

SPIKE_FN="multi"
NEURON_TYPE="iaf"
MIN_VMEM=-1
WEIGHT_INIT="kaiming"

# ---------------------------------------------------------------------------
# Model series
#
# Format: NAME|HIDDEN_DIMS|THRESHOLDS|TEMPORAL_DECAY_STAGES|RESET|TAU_SYN
#
# HIDDEN_DIMS lists hidden layers only. THRESHOLDS is one value (broadcast
# to every spiking layer) or exactly len(HIDDEN_DIMS)+1 values. RESET is
# hard or soft. TAU_SYN is none or an initial synaptic time constant (in
# 4 ms steps, trained per layer).
#
# NAMES ARE CHECKPOINT KEYS: never rename or redefine an existing row; add a
# new one instead. medium_* is the network used for bmi pooled training.
# ---------------------------------------------------------------------------
HKM_TABLE=(
    "small_1x|128|1.0|2|hard|none"
    "medium_2x|256 128|1.25|2|hard|none"
    "medium_2x_soft|256 128|1.25|2|soft|none"
    "medium_2x_tau_syn_8|256 128|1.25|2|hard|8.0"
    "base_3x|256 128 64|1.0|2|hard|none"
    "wide512_2x|512 256|1.0|2|hard|none"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# parse_config ROW -> sets CFG_NAME, CFG_DIMS, CFG_THRESHOLDS, CFG_DECAY,
# CFG_RESET, CFG_TAU_SYN and CFG_EXP_NAME (train_snn.py's file prefix).
parse_config() {
    local thr_spec i t
    local -a dims thr
    IFS='|' read -r CFG_NAME CFG_DIMS thr_spec CFG_DECAY CFG_RESET CFG_TAU_SYN <<< "$1"
    read -ra dims <<< "$CFG_DIMS"
    read -ra thr <<< "$thr_spec"
    local n_expected=$(( ${#dims[@]} + 1 ))

    if [ "${#thr[@]}" -eq 1 ]; then
        t="${thr[0]}"; thr=()
        for (( i = 0; i < n_expected; i++ )); do thr+=("$t"); done
    fi
    if [ "${#thr[@]}" -ne "$n_expected" ]; then
        echo "ERROR: model '$CFG_NAME' has ${#thr[@]} thresholds;" \
             "hidden_dims='$CFG_DIMS' requires $n_expected (hidden + output)." >&2
        return 1
    fi
    if [ "$CFG_RESET" != "hard" ] && [ "$CFG_RESET" != "soft" ]; then
        echo "ERROR: model '$CFG_NAME' has RESET '$CFG_RESET'; use hard or soft." >&2
        return 1
    fi
    CFG_THRESHOLDS="${thr[*]}"
    CFG_EXP_NAME="bmi_${NEURON_TYPE}_${CFG_RESET}"
}

# find_config NAME -> parse_config on the matching row.
find_config() {
    local row
    for row in "${HKM_TABLE[@]}"; do
        if [ "${row%%|*}" = "$1" ]; then
            parse_config "$row"
            return
        fi
    done
    echo "ERROR: model '$1' is not in HKM_TABLE." >&2
    return 1
}

# checkpoint_dir SUBJECT SESSION_ID MODEL
checkpoint_dir() {
    echo "${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/$1/${SERIES_TAG}/$2/$3"
}

# .../snn_datasets/hkm/<subject>/mua/<session> -> <subject>
subject_of() {
    basename "$(dirname "$(dirname "$1")")"
}

# Validate the whole table up front: every row parses, names are unique.
declare -A _seen_names=()
for _row in "${HKM_TABLE[@]}"; do
    parse_config "$_row" || exit 1
    if [ -n "${_seen_names[$CFG_NAME]:-}" ]; then
        echo "ERROR: duplicate model name '$CFG_NAME' in HKM_TABLE." >&2
        exit 1
    fi
    _seen_names[$CFG_NAME]=1
done
unset _row _seen_names

# ---------------------------------------------------------------------------
# PLAN MODE: build a manifest of pending work and print the sbatch command.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--plan" ]; then
    PLAN_SUBJECT="${2:-}"
    glob="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${PLAN_SUBJECT:-*}/${DATASET_SUBDIR}"
    # $glob is intentionally unquoted so the shell expands the subject wildcard.
    mapfile -t session_dirs < <(find $glob -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    if [ "${#session_dirs[@]}" -eq 0 ]; then
        echo "ERROR: no session directories under ${glob}" >&2
        echo "       (run nwb_conversion/run_hkm_nwb_pipeline_array.sbatch first)" >&2
        exit 1
    fi

    # Every subject needs velocity scalers, or every task would fail at startup.
    mapfile -t subjects < <(for d in "${session_dirs[@]}"; do subject_of "$d"; done | sort -u)
    if ! python3 - "$VELOCITY_SCALERS" "$EXPERIMENT" "${subjects[@]}" <<'EOF'
import json, sys
path, experiment, subjects = sys.argv[1], sys.argv[2], sys.argv[3:]
try:
    with open(path) as f:
        scalers = json.load(f).get(experiment, {})
except FileNotFoundError:
    scalers = {}
missing = [s for s in subjects if s not in scalers]
if missing:
    print(f"ERROR: {path} has no {experiment} velocity scalers for: {', '.join(missing)}.\n"
          f"       Run compute_velocity_scalers.py (see this script's header) first.",
          file=sys.stderr)
    sys.exit(1)
EOF
    then
        exit 1
    fi

    mkdir -p snn_logs/manifests
    manifest="$(pwd)/snn_logs/manifests/${SERIES_TAG}_${PLAN_SUBJECT:-all}_$(date +%Y%m%d_%H%M%S)_$$.txt"
    : > "$manifest"

    n_sessions=0; n_done=0; n_partial=0; n_fresh=0; n_filtered=0
    for session_dir in "${session_dirs[@]}"; do
        if [ ! -d "$session_dir/train" ] || [ ! -d "$session_dir/test" ]; then
            echo "WARNING: skipping $session_dir (missing train/ or test/)" >&2
            continue
        fi
        n_sessions=$((n_sessions + 1))
        subject="$(subject_of "$session_dir")"
        session_id="$(basename "$session_dir")"

        for row in "${HKM_TABLE[@]}"; do
            parse_config "$row"
            if [ -n "${HKM_FILTER:-}" ] && ! [[ "$CFG_NAME" =~ $HKM_FILTER ]]; then
                n_filtered=$((n_filtered + 1))
                continue
            fi
            ckpt_dir="$(checkpoint_dir "$subject" "$session_id" "$CFG_NAME")"
            if [ -f "${ckpt_dir}/final_model_${CFG_EXP_NAME}.pth" ]; then
                n_done=$((n_done + 1))
                continue
            fi
            if compgen -G "${ckpt_dir}/checkpoint_${CFG_EXP_NAME}_epoch*.pth" > /dev/null; then
                n_partial=$((n_partial + 1))
            else
                n_fresh=$((n_fresh + 1))
            fi
            echo "${session_dir}|${CFG_NAME}" >> "$manifest"
        done
    done

    n_pending=$(wc -l < "$manifest")
    echo "# subjects:          ${subjects[*]}"
    echo "# sessions:          $n_sessions"
    echo "# models in table:   ${#HKM_TABLE[@]}${HKM_FILTER:+  (filter: $HKM_FILTER, $n_filtered pairs excluded)}"
    echo "# already finished:  $n_done"
    echo "# to resume:         $n_partial"
    echo "# to start fresh:    $n_fresh"
    if [ "$n_pending" -eq 0 ]; then
        rm -f "$manifest"
        echo "# Nothing to do."
        exit 0
    fi
    echo "# manifest:          $manifest"
    echo "mkdir -p snn_logs"
    echo "sbatch --export=ALL,MANIFEST=${manifest} --array=0-$((n_pending - 1)) $0"
    exit 0
fi

# ---------------------------------------------------------------------------
# TASK MODE: run one manifest line.
# ---------------------------------------------------------------------------
if [ -z "${MANIFEST:-}" ] || [ ! -f "$MANIFEST" ]; then
    echo "ERROR: MANIFEST is not set or does not exist. Run with --plan first." >&2
    exit 1
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: this script must run as a Slurm array task." >&2
    exit 1
fi

N_TASKS=$(wc -l < "$MANIFEST")
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$LINE" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no manifest entry ($N_TASKS total); exiting."
    exit 0
fi

IFS='|' read -r SESSION_DIR MODEL_NAME <<< "$LINE"
find_config "$MODEL_NAME"

SUBJECT="$(subject_of "$SESSION_DIR")"
SESSION_ID="$(basename "$SESSION_DIR")"
CHECKPOINT_DIR="$(checkpoint_dir "$SUBJECT" "$SESSION_ID" "$MODEL_NAME")"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${CFG_EXP_NAME}.pth"

if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# Re-check: another job may have finished this since the plan was made.
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] $SUBJECT/$SESSION_ID/$MODEL_NAME already finished: $FINAL_CHECKPOINT"
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${CFG_EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
else
    echo "[fresh] no checkpoint found"
fi

TAU_SYN_ARGS=()
[ "$CFG_TAU_SYN" != "none" ] && TAU_SYN_ARGS=(--tau-syn "$CFG_TAU_SYN")
read -ra HIDDEN_DIMS_ARR <<< "$CFG_DIMS"
read -ra THRESHOLDS_ARR <<< "$CFG_THRESHOLDS"

echo "============================================================"
echo "HKM SNN series: $SERIES_TAG"
echo "============================================================"
echo "Array task:      $SLURM_ARRAY_TASK_ID / $((N_TASKS - 1))"
echo "Manifest:        $MANIFEST"
echo "Subject:         $SUBJECT"
echo "Session:         $SESSION_ID"
echo "Model:           $MODEL_NAME"
echo "Hidden dims:     $CFG_DIMS  (${#HIDDEN_DIMS_ARR[@]} hidden, $(( ${#HIDDEN_DIMS_ARR[@]} + 1 )) weight layers)"
echo "Thresholds:      $CFG_THRESHOLDS"
echo "Temporal decay:  $CFG_DECAY"
echo "Reset:           $CFG_RESET"
echo "Tau syn (init):  $CFG_TAU_SYN"
echo "Batch size:      $BATCH_SIZE (whole trials)"
echo "Epochs / lr:     $EPOCHS / $LR"
echo "Checkpoint dir:  $CHECKPOINT_DIR"
echo "Host:            $(hostname)"
echo "============================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$SESSION_DIR" \
    --experiment "$EXPERIMENT" \
    --config "$CONFIG_YAML" \
    --spike-fn "$SPIKE_FN" \
    --min-vmem "$MIN_VMEM" \
    --use-iaf-squeeze \
    --weight-init "$WEIGHT_INIT" \
    --neuron-type "$NEURON_TYPE" \
    --reset-type "$CFG_RESET" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    --scheduler-patience "$SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    "${TAU_SYN_ARGS[@]}" \
    --temporal-decay-stages "$CFG_DECAY" \
    --hidden-dims "${HIDDEN_DIMS_ARR[@]}" \
    --spike-thresholds "${THRESHOLDS_ARR[@]}" \
    "${RESUME_ARGS[@]}"
