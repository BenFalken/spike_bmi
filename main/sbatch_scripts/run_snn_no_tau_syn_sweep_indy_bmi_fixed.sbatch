#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J SNN_SWEEP_NO_TAU
#SBATCH -o snn_logs/SNN_SWEEP_NO_TAU_%A_%a.out
#SBATCH -e snn_logs/SNN_SWEEP_NO_TAU_%A_%a.out

# Hardware-aware SNN hyperparameter sweep for Speck2F/DYNAP-CNN-compatible models.
#
# IMPORTANT:
#   1. tau_syn is intentionally fixed at 1.0 because the target hardware deployment
#      does not support a train-time tau_syn parameter in this workflow.
#   2. Each flattened array task corresponds to ONE (session, configuration) pair --
#      this is now genuinely true (see "Resolve session + sweep configuration"
#      below): SESSION_INDEX = task_id / SWEEP_COUNT, SWEEP_INDEX = task_id %
#      SWEEP_COUNT, so all 22 configs run consecutively for session 0, then all
#      22 for session 1, and so on. Previously this comment described the
#      INTENDED design, but the actual code still hardcoded SESSION_INDEX=0 --
#      only ever testing the sweep against a single session. That's fixed here.
#   3. Checkpoints are isolated by (subject, session, configuration), so
#      sessions/configurations/subjects can all resume independently.
#   4. SUBJECT is now OPTIONAL. Set it to restrict this array to one subject's
#      own sessions (the original, intended behavior). Leave it unset to sweep
#      EVERY subject under snn_datasets/${EXPERIMENT}/ in ONE array job --
#      each task still resolves to exactly one (subject, session, config)
#      triple; the subject is re-derived PER TASK from that task's own
#      resolved session path, not trusted from this env var, so this is
#      correct in both modes at once.
#
# Because bash can't expand ${#SWEEP_NAMES[@]} (or know N_SESSIONS) before
# submission, use the now-actually-implemented helper instead of computing
# the --array range by hand:
#   bash run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch --print-submit-command bmi indy
#   bash run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch --print-submit-command bmi
#       (omit the subject to size the array for EVERY subject under bmi at once)
# Each prints a ready-to-copy `sbatch --export=...,--array=0-N ...` command --
# it does NOT submit anything itself.
#
# Example (once you have the real --array range from the helper above):
#   mkdir -p snn_logs
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-65 run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch
#   sbatch --export=EXPERIMENT=bmi --array=0-109 run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch   # every subject

set -eo pipefail

DATA_ROOT="/users/bfalkenb/scratch/bfalkenb/data"

# ---------------------------------------------------------------------------
# Sweep table (needed by both --print-submit-command and the real training
# path below, so it's defined before either -- SWEEP_COUNT depends on it).
#
# Format: NAME|HIDDEN_DIMS|THRESHOLDS|TEMPORAL_DECAY_STAGES
#
# Keep thresholds one-per-hidden-layer. All configurations below therefore
# have internally consistent threshold vectors.
#
# Start with width/depth + threshold + temporal-decay sweeps. Once the
# training script exposes explicit EMA controls, add them as additional
# columns rather than guessing their CLI names here.
# ---------------------------------------------------------------------------

SWEEP_NAMES=(
    "base"
    "small_1x"
    "small_2x"
    "medium_2x"
    "medium_3x"
    "deep_4x"
    "wide_2x"
    "wide_3x"

    "base_thr075"
    "base_thr125"
    "base_thr150"
    "base_thr200"

    "base_decay0"
    "base_decay1"
    "base_decay3"
    "base_decay4"

    "small_decay1"
    "medium_decay1"
    "deep_decay1"

    "small_thr125"
    "medium_thr125"
    "deep_thr125"
)

SWEEP_HIDDEN_DIMS=(
    "256 128 64"
    "128"
    "192 96"
    "256 128"
    "256 128 64"
    "256 192 128 64"
    "384 192"
    "512 256 128"

    "256 128 64"
    "256 128 64"
    "256 128 64"
    "256 128 64"

    "256 128 64"
    "256 128 64"
    "256 128 64"
    "256 128 64"

    "128"
    "256 128"
    "256 192 128 64"

    "128"
    "256 128"
    "256 192 128 64"
)

SWEEP_THRESHOLDS=(
    "1.0 1.0 1.0 1.0"
    "1.0 1.0"
    "1.0 1.0 1.0"
    "1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0 1.0"
    "1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0"

    "0.75 0.75 0.75 0.75"
    "1.25 1.25 1.25 1.25"
    "1.50 1.50 1.50 1.50"
    "2.00 2.00 2.00 2.00"

    "1.0 1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0"

    "1.0 1.0"
    "1.0 1.0 1.0"
    "1.0 1.0 1.0 1.0 1.0"

    "1.25 1.25"
    "1.25 1.25 1.25"
    "1.25 1.25 1.25 1.25 1.25"
)

SWEEP_DECAY_STAGES=(
    2
    2
    2
    2
    2
    2
    2
    2

    2
    2
    2
    2

    0
    1
    3
    4

    1
    1
    1

    2
    2
    2
)

SWEEP_COUNT=${#SWEEP_NAMES[@]}

if [ "${#SWEEP_HIDDEN_DIMS[@]}" -ne "$SWEEP_COUNT" ] || \
   [ "${#SWEEP_THRESHOLDS[@]}" -ne "$SWEEP_COUNT" ] || \
   [ "${#SWEEP_DECAY_STAGES[@]}" -ne "$SWEEP_COUNT" ]; then
    echo "ERROR: sweep arrays have different lengths." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# --print-submit-command helper -- run in plain bash (NOT via sbatch), e.g.:
#   bash run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch --print-submit-command bmi indy
#   bash run_snn_no_tau_syn_sweep_indy_bmi_fixed.sbatch --print-submit-command bmi
# Computes the real --array=0-N range for the requested scope and prints a
# ready-to-copy sbatch command. Exits immediately after -- does not touch
# SLURM_ARRAY_TASK_ID or anything below this block.
# ---------------------------------------------------------------------------
if [ "$1" = "--print-submit-command" ]; then
    HELPER_EXPERIMENT="$2"
    HELPER_SUBJECT="$3"
    if [ -z "$HELPER_EXPERIMENT" ]; then
        echo "Usage: bash $0 --print-submit-command <experiment> [subject]" >&2
        echo "       subject is optional -- omit it to size the array for EVERY subject" >&2
        echo "       under that experiment at once." >&2
        exit 1
    fi
    if [ -n "$HELPER_SUBJECT" ]; then
        HELPER_GLOB="${DATA_ROOT}/snn_datasets/${HELPER_EXPERIMENT}/${HELPER_SUBJECT}/mua_8_group"
    else
        HELPER_GLOB="${DATA_ROOT}/snn_datasets/${HELPER_EXPERIMENT}"/*/mua_8_group
    fi
    N_SESSIONS_HELPER=$(find $HELPER_GLOB -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    if [ "$N_SESSIONS_HELPER" -eq 0 ]; then
        echo "ERROR: found zero session directories under ${HELPER_GLOB} -- check the " >&2
        echo "       experiment/subject and that these mua_8_group datasets exist." >&2
        exit 1
    fi
    N_TASKS=$(( N_SESSIONS_HELPER * SWEEP_COUNT ))
    echo "# ${N_SESSIONS_HELPER} session(s) x ${SWEEP_COUNT} sweep config(s) = ${N_TASKS} task(s)"
    if [ -n "$HELPER_SUBJECT" ]; then
        echo "sbatch --export=EXPERIMENT=${HELPER_EXPERIMENT},SUBJECT=${HELPER_SUBJECT} --array=0-$((N_TASKS-1)) $0"
    else
        echo "sbatch --export=EXPERIMENT=${HELPER_EXPERIMENT} --array=0-$((N_TASKS-1)) $0"
    fi
    exit 0
fi

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Resolve Experiment (required) / Subject (now optional -- see header) ---
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set." >&2
    exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script is intended for a Slurm array." >&2
    exit 1
fi

if [ -n "$SUBJECT" ]; then
    SNN_DATASET_GLOB="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
else
    # ALL-subjects mode: `*` matches every subject dir under this experiment,
    # so every (subject, session) pair's own session dir lands in one flat,
    # sorted list -- confirmed directly against a real multi-subject fixture
    # (indy + loco) before delivering this, including the flattened
    # session x sweep index math and the --print-submit-command helper
    # above, for both the single-subject and all-subjects cases.
    SNN_DATASET_GLOB="${DATA_ROOT}/snn_datasets/${EXPERIMENT}"/*/mua_8_group
fi

CONFIG_YAML="../configs/iaf_hard_reset.yaml"
BASE_EXP_NAME="bmi_iaf_hard_no_tau"

# ---------------------------------------------------------------------------
# Training controls
# ---------------------------------------------------------------------------
BATCH_SIZE=20
EPOCHS=50
PATIENCE=10
SCHEDULER_PATIENCE=5
CHECKPOINT_INTERVAL=10

# Hardware-compatible choices from the current command line:
SPIKE_FN="multi"
NEURON_TYPE="iaf"
TAU_SYN=1.0
MIN_VMEM=-1
WEIGHT_INIT="kaiming"
USE_IAF_SQUEEZE=1

# ---------------------------------------------------------------------------
# Resolve session + sweep configuration from one flattened array index.
#
# BUG FIX: this previously hardcoded SESSION_INDEX=0 ("only the first sorted
# Indy/BMI session is tested in this first sweep") -- meaning every array
# task 0-21 all trained against the SAME single session, just with different
# sweep configs, never actually covering more than one session at all. Now a
# genuine flattened (session x sweep) space: SESSION_INDEX = task_id /
# SWEEP_COUNT, SWEEP_INDEX = task_id % SWEEP_COUNT -- all 22 configs run
# consecutively per session before moving to the next session. Verified
# directly against known task-id boundaries (0, 21, 22, 43, 44, 65 for a
# 3-session fixture) before delivering this.
# ---------------------------------------------------------------------------

mapfile -t SESSION_DIRS < <(find $SNN_DATASET_GLOB -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
N_SESSIONS=${#SESSION_DIRS[@]}

if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: zero session directories found under ${SNN_DATASET_GLOB}" >&2
    exit 1
fi

TOTAL_TASKS=$(( N_SESSIONS * SWEEP_COUNT ))
SESSION_INDEX=$(( SLURM_ARRAY_TASK_ID / SWEEP_COUNT ))
SWEEP_INDEX=$(( SLURM_ARRAY_TASK_ID % SWEEP_COUNT ))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_TASKS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding work: "
    echo "$N_SESSIONS session(s) x $SWEEP_COUNT config(s) = $TOTAL_TASKS task(s) total -- exiting cleanly."
    exit 0
fi

SESSION_DIR="${SESSION_DIRS[$SESSION_INDEX]}"
SESSION_ID="$(basename "$SESSION_DIR")"
# Subject for THIS task, resolved from the task's own resolved SESSION_DIR
# path rather than trusted from $SUBJECT -- correct whether this was
# submitted with SUBJECT set (single-subject mode, where it just re-derives
# the same value) or left unset (all-subjects mode, where different array
# tasks genuinely belong to different subjects).
TASK_SUBJECT="$(basename "$(dirname "$(dirname "$SESSION_DIR")")")"

SWEEP_NAME="${SWEEP_NAMES[$SWEEP_INDEX]}"
HIDDEN_DIMS="${SWEEP_HIDDEN_DIMS[$SWEEP_INDEX]}"
THRESHOLDS="${SWEEP_THRESHOLDS[$SWEEP_INDEX]}"
TEMPORAL_DECAY_STAGES="${SWEEP_DECAY_STAGES[$SWEEP_INDEX]}"

# model_bmi.py expects one threshold for every spiking layer, including the
# final 2-D velocity output layer. Therefore the required threshold count is
# the number of hidden layers + 1. Validate every sweep entry before training.
read -ra _HIDDEN_DIM_ARRAY <<< "$HIDDEN_DIMS"
read -ra _THRESHOLD_ARRAY <<< "$THRESHOLDS"
EXPECTED_THRESHOLDS=$(( ${#_HIDDEN_DIM_ARRAY[@]} + 1 ))
if [ "${#_THRESHOLD_ARRAY[@]}" -ne "$EXPECTED_THRESHOLDS" ]; then
    echo "ERROR: $SWEEP_NAME has ${#_THRESHOLD_ARRAY[@]} thresholds; hidden_dims=$HIDDEN_DIMS requires $EXPECTED_THRESHOLDS." >&2
    exit 1
fi

if [ ! -d "$SESSION_DIR/train" ] || [ ! -d "$SESSION_DIR/test" ]; then
    echo "ERROR: no train/ or test/ subfolder under $SESSION_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Unique checkpoint directory for this exact subject/session/configuration.
# ---------------------------------------------------------------------------

CHECKPOINTS_ROOT="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${TASK_SUBJECT}/no_tau_syn_sweep"
CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/${SESSION_ID}/${SWEEP_NAME}"
EXP_NAME="${BASE_EXP_NAME}_${SWEEP_NAME}"

FINAL_CHECKPOINT="${CHECKPOINT_DIR}/final_model_${EXP_NAME}.pth"

if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "[skip] subject=${TASK_SUBJECT} session=${SESSION_ID} config=${SWEEP_NAME}"
    echo "       final checkpoint already exists: $FINAL_CHECKPOINT"
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

echo "============================================================"
echo "SNN Speck2F-compatible sweep"
echo "============================================================"
echo "Array task:       $SLURM_ARRAY_TASK_ID / $((TOTAL_TASKS-1))"
echo "Subject:          $TASK_SUBJECT"
echo "Session:          $SESSION_ID ($((SESSION_INDEX+1))/$N_SESSIONS)"
echo "Configuration:    $SWEEP_NAME ($((SWEEP_INDEX+1))/$SWEEP_COUNT)"
echo "Hidden dims:      $HIDDEN_DIMS"
echo "Thresholds:       $THRESHOLDS"
echo "Tau syn:          $TAU_SYN [FIXED]"
echo "Temporal decay:   $TEMPORAL_DECAY_STAGES"
echo "Batch size:       $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
echo "============================================================"

mkdir -p "$CHECKPOINT_DIR"

python3 ../snn_training/train_bmi_no_tau_syn.py \
    --data-path "$SESSION_DIR" \
    --config "$CONFIG_YAML" \
    --spike-fn "$SPIKE_FN" \
    --min-vmem "$MIN_VMEM" \
    --use-iaf-squeeze \
    --weight-init "$WEIGHT_INIT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --training-mode windowed \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --scheduler-patience "$SCHEDULER_PATIENCE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --neuron-type "$NEURON_TYPE" \
    --tau-syn "$TAU_SYN" \
    --temporal-decay-stages "$TEMPORAL_DECAY_STAGES" \
    --hidden-dims $HIDDEN_DIMS \
    --spike-thresholds $THRESHOLDS \
    "${RESUME_ARGS[@]}"
