#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_SWEEP
#SBATCH -o snn_logs/SNN_SWEEP_%A_%a.out
#SBATCH -e snn_logs/SNN_SWEEP_%A_%a.out
#
# SNN architecture/hyperparameter sweep: every config in SWEEP_TABLE is
# trained separately on every session's grouped dataset (mua_8_group).
#
# TAU_SYN selects the sweep variant:
#   TAU_SYN=none (default)  no synaptic-current stage -> <subject>/no_tau_syn_sweep/
#   TAU_SYN=8.0             tau_syn initialized at 8.0 and trained -> <subject>/tau_syn_8.0_sweep/
# It only needs to be set when planning; the manifest records it.
#
# Two-step workflow
# -----------------
# 1. PLAN (plain bash, from the directory you will submit from):
#
#      bash run_snn_sweep.sbatch --plan bmi                      # every subject
#      bash run_snn_sweep.sbatch --plan bmi indy                 # one subject
#      TAU_SYN=8.0 bash run_snn_sweep.sbatch --plan bmi indy     # tau_syn = 8.0 variant
#      SWEEP_FILTER='^wide' bash run_snn_sweep.sbatch --plan bmi
#
#    This enumerates every (subject, session, config) triple, drops every
#    triple that already has a final checkpoint, writes the rest to a
#    manifest under snn_logs/manifests/, and prints a ready-to-copy sbatch
#    command. It submits nothing. SWEEP_FILTER is an optional bash regex
#    matched against config names.
#
# 2. SUBMIT the printed command, e.g.
#      sbatch --export=ALL,MANIFEST=/abs/path/manifest.txt --array=0-219 run_snn_sweep.sbatch
#
# Array task i runs line i of the manifest. Each task re-checks for a final
# checkpoint (skip) and resumes from its latest epoch checkpoint if present,
# so after timeouts you can simply re-plan and re-submit.
#
# Outputs, one directory per (session, config):
#   ${DATA_ROOT}/snn_checkpoints/<exp>/<subject>/<sweep tag>/<session>/<config>/
#       final_model_bmi_iaf_hard.pth, best_model_weights.pth,
#       checkpoint_bmi_iaf_hard_epoch<N>.pth
# Summarize with plot_snn_sweep_heatmap.py --checkpoint-root <.../sweep tag>.

set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
DATASET_SUBDIR="mua_8_group"
CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
# File prefix train_snn.py uses for an IAF / hard-reset model.
EXP_NAME="bmi_iaf_hard"

# set_tau_syn VALUE -> sets TAU_SYN, SWEEP_TAG (checkpoint subdirectory) and TAU_SYN_ARGS.
set_tau_syn() {
    TAU_SYN="$1"
    if [ "$TAU_SYN" = "none" ]; then
        SWEEP_TAG="no_tau_syn_sweep"
        TAU_SYN_ARGS=()
    else
        SWEEP_TAG="tau_syn_${TAU_SYN}_sweep"
        TAU_SYN_ARGS=(--tau-syn "$TAU_SYN")
    fi
}
set_tau_syn "${TAU_SYN:-none}"

# ---------------------------------------------------------------------------
# Training controls (shared by every config)
# ---------------------------------------------------------------------------
BATCH_SIZE=20
EPOCHS=50
PATIENCE=10
SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=10

SPIKE_FN="multi"
NEURON_TYPE="iaf"
MIN_VMEM=-1
WEIGHT_INIT="kaiming"

# ---------------------------------------------------------------------------
# Sweep table
#
# Format: NAME|HIDDEN_DIMS|THRESHOLDS|TEMPORAL_DECAY_STAGES
#
# HIDDEN_DIMS lists hidden layers only (input and output sizes come from the
#   data/model). "Layers" in names counts weight layers = hidden + 1, e.g.
#   96 -> 192 -> 256 -> 256 -> 192 -> 96 -> out  is a 6-layer net (5 hidden).
# THRESHOLDS is either one value (broadcast to every spiking layer, i.e.
#   hidden + output) or an explicit vector of exactly len(HIDDEN_DIMS)+1.
#
# NAMES ARE CHECKPOINT KEYS: never rename or redefine an existing entry;
# add a new row instead. Order does not matter.
# ---------------------------------------------------------------------------
SWEEP_TABLE=(
    # ----- Original sweep (completed for bmi/indy and bmi/loco) -----
    "base|256 128 64|1.0|2"
    "small_1x|128|1.0|2"
    "small_2x|192 96|1.0|2"
    "medium_2x|256 128|1.0|2"
    "medium_3x|256 128 64|1.0|2"
    "deep_4x|256 192 128 64|1.0|2"
    "wide_2x|384 192|1.0|2"
    "wide_3x|512 256 128|1.0|2"

    "base_thr075|256 128 64|0.75|2"
    "base_thr125|256 128 64|1.25|2"
    "base_thr150|256 128 64|1.50|2"
    "base_thr200|256 128 64|2.00|2"

    "base_decay0|256 128 64|1.0|0"
    "base_decay1|256 128 64|1.0|1"
    "base_decay3|256 128 64|1.0|3"
    "base_decay4|256 128 64|1.0|4"

    "small_decay1|128|1.0|1"
    "medium_decay1|256 128|1.0|1"
    "deep_decay1|256 192 128 64|1.0|1"

    "small_thr125|128|1.25|2"
    "medium_thr125|256 128|1.25|2"
    "deep_thr125|256 192 128 64|1.25|2"

    # ----- 6-layer nets (5 hidden), expand-then-contract -----
    "deep6_small|128 192 192 128 64|1.0|2"
    "deep6_base|192 256 256 192 96|1.0|2"
    "deep6_wide|256 384 384 256 128|1.0|2"

    "deep6_base_thr075|192 256 256 192 96|0.75|2"
    "deep6_base_thr125|192 256 256 192 96|1.25|2"
    "deep6_base_thr150|192 256 256 192 96|1.50|2"
    # Graded: higher early, lower late, to counter spike attenuation with depth.
    "deep6_base_thrtaper|192 256 256 192 96|1.25 1.25 1.0 1.0 0.75 0.75|2"

    "deep6_base_decay1|192 256 256 192 96|1.0|1"
    "deep6_base_decay3|192 256 256 192 96|1.0|3"

    "deep6_small_thr125|128 192 192 128 64|1.25|2"
    "deep6_wide_thr125|256 384 384 256 128|1.25|2"

    # ----- 8-layer nets (7 hidden), expand-plateau-contract -----
    "deep8_small|128 160 192 192 160 128 64|1.0|2"
    "deep8_base|192 224 256 256 224 192 96|1.0|2"
    "deep8_wide|256 320 384 384 320 256 128|1.0|2"

    "deep8_base_thr075|192 224 256 256 224 192 96|0.75|2"
    "deep8_base_thr125|192 224 256 256 224 192 96|1.25|2"
    "deep8_base_thr150|192 224 256 256 224 192 96|1.50|2"
    "deep8_base_thrtaper|192 224 256 256 224 192 96|1.25 1.25 1.0 1.0 1.0 1.0 0.75 0.75|2"

    "deep8_base_decay1|192 224 256 256 224 192 96|1.0|1"
    "deep8_base_decay3|192 224 256 256 224 192 96|1.0|3"

    "deep8_small_thr125|128 160 192 192 160 128 64|1.25|2"
    "deep8_wide_thr125|256 320 384 384 320 256 128|1.25|2"

    # ----- Wide, shallow nets beyond 512 units -----
    # Naming: wide<first hidden width>_<hidden layers>x; each layer halves
    # the previous one unless marked _flat.
    "wide512_1x|512|1.0|2"
    "wide512_2x|512 256|1.0|2"
    "wide768_1x|768|1.0|2"
    "wide768_2x|768 384|1.0|2"
    "wide768_3x|768 384 192|1.0|2"
    "wide768_2x_flat|768 768|1.0|2"
    "wide1024_1x|1024|1.0|2"
    "wide1024_2x|1024 512|1.0|2"
    "wide1024_3x|1024 512 256|1.0|2"

    "wide768_2x_thr125|768 384|1.25|2"
    "wide1024_2x_thr125|1024 512|1.25|2"
    "wide768_2x_decay3|768 384|1.0|3"
    "wide1024_2x_decay3|1024 512|1.0|3"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# parse_config ROW -> sets CFG_NAME, CFG_DIMS, CFG_THRESHOLDS, CFG_DECAY.
# Broadcasts a scalar threshold and validates the threshold count.
parse_config() {
    local thr_spec i t
    local -a dims thr
    IFS='|' read -r CFG_NAME CFG_DIMS thr_spec CFG_DECAY <<< "$1"
    read -ra dims <<< "$CFG_DIMS"
    read -ra thr <<< "$thr_spec"
    local n_expected=$(( ${#dims[@]} + 1 ))

    if [ "${#thr[@]}" -eq 1 ]; then
        t="${thr[0]}"; thr=()
        for (( i = 0; i < n_expected; i++ )); do thr+=("$t"); done
    fi
    if [ "${#thr[@]}" -ne "$n_expected" ]; then
        echo "ERROR: config '$CFG_NAME' has ${#thr[@]} thresholds;" \
             "hidden_dims='$CFG_DIMS' requires $n_expected (hidden + output)." >&2
        return 1
    fi
    CFG_THRESHOLDS="${thr[*]}"
}

# find_config NAME -> parse_config on the matching row.
find_config() {
    local row
    for row in "${SWEEP_TABLE[@]}"; do
        if [ "${row%%|*}" = "$1" ]; then
            parse_config "$row"
            return
        fi
    done
    echo "ERROR: config '$1' is not in SWEEP_TABLE." >&2
    return 1
}

# checkpoint_dir EXPERIMENT SUBJECT SESSION_ID CONFIG
checkpoint_dir() {
    echo "${DATA_ROOT}/snn_checkpoints/$1/$2/${SWEEP_TAG}/$3/$4"
}

# subject is the directory two levels above the session dir:
#   .../snn_datasets/<exp>/<subject>/mua_8_group/<session>
subject_of() {
    basename "$(dirname "$(dirname "$1")")"
}

# Validate the whole table up front: every row parses, names are unique.
declare -A _seen_names=()
for _row in "${SWEEP_TABLE[@]}"; do
    parse_config "$_row" || exit 1
    if [ -n "${_seen_names[$CFG_NAME]:-}" ]; then
        echo "ERROR: duplicate config name '$CFG_NAME' in SWEEP_TABLE." >&2
        exit 1
    fi
    _seen_names[$CFG_NAME]=1
done
unset _row _seen_names

# ---------------------------------------------------------------------------
# PLAN MODE: build a manifest of pending work and print the sbatch command.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--plan" ]; then
    PLAN_EXPERIMENT="${2:-}"
    PLAN_SUBJECT="${3:-}"
    if [ -z "$PLAN_EXPERIMENT" ]; then
        echo "Usage: [SWEEP_FILTER=regex] bash $0 --plan <experiment> [subject]" >&2
        exit 1
    fi

    if [ -n "$PLAN_SUBJECT" ]; then
        glob="${DATA_ROOT}/snn_datasets/${PLAN_EXPERIMENT}/${PLAN_SUBJECT}/${DATASET_SUBDIR}"
    else
        glob="${DATA_ROOT}/snn_datasets/${PLAN_EXPERIMENT}/*/${DATASET_SUBDIR}"
    fi
    # $glob is intentionally unquoted so the shell expands the subject wildcard.
    mapfile -t session_dirs < <(find $glob -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    if [ "${#session_dirs[@]}" -eq 0 ]; then
        echo "ERROR: no session directories under ${glob}" >&2
        exit 1
    fi

    mkdir -p snn_logs/manifests
    manifest="$(pwd)/snn_logs/manifests/${SWEEP_TAG}_${PLAN_EXPERIMENT}_${PLAN_SUBJECT:-all}_$(date +%Y%m%d_%H%M%S)_$$.txt"
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

        for row in "${SWEEP_TABLE[@]}"; do
            parse_config "$row"
            if [ -n "${SWEEP_FILTER:-}" ] && ! [[ "$CFG_NAME" =~ $SWEEP_FILTER ]]; then
                n_filtered=$((n_filtered + 1))
                continue
            fi
            ckpt_dir="$(checkpoint_dir "$PLAN_EXPERIMENT" "$subject" "$session_id" "$CFG_NAME")"
            if [ -f "${ckpt_dir}/final_model_${EXP_NAME}.pth" ]; then
                n_done=$((n_done + 1))
                continue
            fi
            if compgen -G "${ckpt_dir}/checkpoint_${EXP_NAME}_epoch*.pth" > /dev/null; then
                n_partial=$((n_partial + 1))
            else
                n_fresh=$((n_fresh + 1))
            fi
            echo "${PLAN_EXPERIMENT}|${session_dir}|${CFG_NAME}|${TAU_SYN}" >> "$manifest"
        done
    done

    n_pending=$(wc -l < "$manifest")
    echo "# sweep:             $SWEEP_TAG (TAU_SYN=$TAU_SYN)"
    echo "# sessions:          $n_sessions"
    echo "# configs in table:  ${#SWEEP_TABLE[@]}${SWEEP_FILTER:+  (filter: $SWEEP_FILTER, $n_filtered triples excluded)}"
    echo "# already finished:  $n_done"
    echo "# to resume:         $n_partial"
    echo "# to start fresh:    $n_fresh"
    if [ "$n_pending" -eq 0 ]; then
        rm -f "$manifest"
        echo "# Nothing to do."
        exit 0
    fi
    echo "# manifest:          $manifest"
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

# Manifest lines are EXPERIMENT|SESSION_DIR|CONFIG|TAU_SYN; the manifest's
# TAU_SYN wins over the environment so a task always matches its plan.
IFS='|' read -r EXPERIMENT SESSION_DIR SWEEP_NAME MANIFEST_TAU_SYN <<< "$LINE"
set_tau_syn "${MANIFEST_TAU_SYN:-none}"
find_config "$SWEEP_NAME"

SUBJECT="$(subject_of "$SESSION_DIR")"
SESSION_ID="$(basename "$SESSION_DIR")"
CHECKPOINT_DIR="$(checkpoint_dir "$EXPERIMENT" "$SUBJECT" "$SESSION_ID" "$SWEEP_NAME")"
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth"

if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# Re-check: another job may have finished this since the plan was made.
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] $EXPERIMENT/$SUBJECT/$SESSION_ID/$SWEEP_NAME already finished: $FINAL_CHECKPOINT"
    exit 0
fi

RESUME_ARGS=()
LATEST_CHECKPOINT=$(ls "${CHECKPOINT_DIR}"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "[resume] $LATEST_CHECKPOINT"
    RESUME_ARGS+=(--resume "$LATEST_CHECKPOINT")
else
    echo "[fresh] no checkpoint found"
fi

read -ra HIDDEN_DIMS_ARR <<< "$CFG_DIMS"
read -ra THRESHOLDS_ARR <<< "$CFG_THRESHOLDS"

echo "============================================================"
echo "SNN sweep: $SWEEP_TAG"
echo "============================================================"
echo "Array task:      $SLURM_ARRAY_TASK_ID / $((N_TASKS - 1))"
echo "Manifest:        $MANIFEST"
echo "Experiment:      $EXPERIMENT"
echo "Subject:         $SUBJECT"
echo "Session:         $SESSION_ID"
echo "Configuration:   $SWEEP_NAME"
echo "Hidden dims:     $CFG_DIMS  (${#HIDDEN_DIMS_ARR[@]} hidden, $(( ${#HIDDEN_DIMS_ARR[@]} + 1 )) weight layers)"
echo "Thresholds:      $CFG_THRESHOLDS"
echo "Tau syn (init):  $TAU_SYN"
echo "Temporal decay:  $CFG_DECAY"
echo "Batch size:      $BATCH_SIZE"
echo "Epochs:          $EPOCHS"
echo "Checkpoint dir:  $CHECKPOINT_DIR"
echo "============================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$CHECKPOINT_DIR"

python3 "$TRAIN_SCRIPT" \
    --data-path "$SESSION_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn "$SPIKE_FN" \
    --min-vmem "$MIN_VMEM" \
    --use-iaf-squeeze \
    --weight-init "$WEIGHT_INIT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --scheduler-patience "$SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type "$NEURON_TYPE" \
    "${TAU_SYN_ARGS[@]}" \
    --temporal-decay-stages "$CFG_DECAY" \
    --hidden-dims "${HIDDEN_DIMS_ARR[@]}" \
    --spike-thresholds "${THRESHOLDS_ARR[@]}" \
    "${RESUME_ARGS[@]}"
