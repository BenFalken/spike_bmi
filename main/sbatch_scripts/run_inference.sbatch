#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J INFERENCE
#SBATCH -o logs/INFERENCE_%A_%a.out
#SBATCH -e logs/INFERENCE_%A_%a.out
#
# Evaluate every decoder on every session of one subject, then build the report.
#
#   bash run_inference.sbatch --submit bmi indy     # on the cluster
#   bash run_inference.sbatch --local bmi indy      # on one machine, e.g. the Speck laptop
#   bash run_inference.sbatch --local bmi           # every subject of the experiment in turn
#
# Without a subject, every subject with a ${DATA_ROOT}/snn_checkpoints/<experiment>/
# <subject>/per_session/ directory is run as if named (one report per subject).
#
# --submit submits (from this directory):
#   1. an array job, one task per session with an SNN checkpoint, running
#      inference/test_all_decoders.py: accuracy (full data and per training
#      duration) and latency / energy profiling -> sessions/<session>.json
#   2. a report job that starts when the array has finished, running
#      inference/make_report.py: combined_metrics*.json, efficiency_summary.json
#      and the efficiency, energy and 4x2 figures
# all under ${DATA_ROOT}/results/test_all_decoders/<experiment>/<subject>/.
#
# --local runs the same steps in series without Slurm. Its defaults suit the
# Speck-connected laptop: DECODERS adds 'speck' (the SNN on the devkit),
# TRAIN_DURATIONS is empty (durations come from the cluster run), MACHINE=laptop
# labels its profile, and math libraries use one thread. The data layout under
# DATA_ROOT is the same as on the cluster; to extend the cluster results, copy
# the cluster's sessions/*.json into the laptop's results directory first: their
# durations and cluster profile are kept, and full-data results gain 'speck'.
# The laptop needs only the test split of each dataset/.../<session>_binning.h5
# (inference/export_test_split.py writes those), the full-data model bundles,
# the SNN checkpoints and snn_datasets/.../<session>/test/.
#
# Re-running is safe: finished session sections are skipped, and the report is
# always rebuilt. Settings below can be given as NAME=value after the subject,
#   bash run_inference.sbatch --local bmi indy SNN_CHECKPOINT_ROOT=.../per_session FIGURES=1
# or in the environment: before the command on the same line, or exported
# (a plain NAME=value on a line of its own is not seen by this script), e.g.
#   SNN_CHECKPOINT_ROOT=.../full_cohort_finetuned_medium bash run_inference.sbatch --submit bmi indy
#   SPECK_CHECKPOINT_ROOT=.../per_session ...   run 'speck' with other checkpoints than 'snn'
#   FIGURES=1 bash run_inference.sbatch --submit bmi indy     # also per-session diagnostic figures
#   FIGURE_DECODERS=snn,speck FIGURES=1 ...   only these decoders in them (and the crosshair GIFs)
#   DATA_ROOT=~/data VENV=~/speck-env bash run_inference.sbatch --local bmi indy
#   (--local and --report without DATA_ROOT use the data/ folder next to main/, if there is one)
#   SPECK_WAIT_TIME=0.005 ...   seconds the chip gets per timestep before its output is read
#   Only rebuild the report:  bash run_inference.sbatch --report bmi indy
#   (training-duration results missing from sessions/*.json are kept from the
#   existing combined_metrics_durations.json, or taken from DURATIONS_JSON=...)
#   Only redraw the per-session figures and crosshair GIFs from the saved
#   predictions (no evaluation, no chip, any checkpoints):
#       bash run_inference.sbatch --figures bmi indy FIGURE_DECODERS=snn,speck

set -eo pipefail

# Settings this script reads from the environment (see the examples above).
SETTINGS="DATA_ROOT VENV DECODERS TRAIN_DURATIONS MACHINE SNN_CHECKPOINT_ROOT SNN_CHECKPOINT_SUBDIR
          SPECK_CHECKPOINT_ROOT SPECK_CHECKPOINT_SUBDIR RESULTS_DIR FIGURES FIGURE_DECODERS
          SPECK_WAIT_TIME ENERGY_METER_NUM_THREADS DURATIONS_JSON"

case "${1:-}" in
    --submit|--local|--report|--figures)
        case "$1" in
            --report) MODE=report ;;
            --figures) MODE=figures ;;
            *) RUN="$1" ;;
        esac
        shift
        POSITIONAL=()
        for arg in "$@"; do
            if [[ "$arg" == *=* ]]; then          # NAME=value: same as setting it in the environment
                name="${arg%%=*}"; value="${arg#*=}"
                if [[ " $(echo $SETTINGS) " != *" $name "* ]]; then
                    echo "ERROR: unknown setting '$name' (known: $(echo $SETTINGS))" >&2
                    exit 1
                fi
                [[ "$value" == "~/"* ]] && value="$HOME/${value#\~/}"
                export "$name=$value"
            else
                POSITIONAL+=("$arg")
            fi
        done
        EXPERIMENT="${POSITIONAL[0]:-}"; SUBJECT="${POSITIONAL[1]:-}" ;;
esac
if [ -z "${EXPERIMENT:-}" ] || { [ -z "${SUBJECT:-}" ] && [ -z "${RUN:-}" ]; }; then
    echo "Usage: bash $0 --submit|--local <experiment> [<subject>] [NAME=value ...]" >&2
    echo "       bash $0 --report|--figures <experiment> <subject> [NAME=value ...]" >&2
    exit 1
fi
# Without a mode flag the script is one Slurm array task (MODE=session, set by
# --submit); outside Slurm that would quietly run only the first session.
if [ -z "${RUN:-}" ] && [ -z "${MODE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: no mode given. To run every session here: bash $0 --local ${EXPERIMENT} ${SUBJECT:-} [NAME=value ...]" >&2
    echo "       (or --submit for Slurm, --report / --figures for the report or figures only)" >&2
    exit 1
fi
# Data root: $DATA_ROOT if set; outside Slurm (--local, --report), else the
# data/ folder next to main/ (repo/main/../data) when it exists; else the
# cluster scratch directory.
LOCAL_DATA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/data"
if [ -z "${DATA_ROOT:-}" ] && [ "${RUN:-}" != "--submit" ] && [ -z "${SLURM_JOB_ID:-}" ] \
        && [ -d "$LOCAL_DATA_ROOT" ]; then
    DATA_ROOT="$LOCAL_DATA_ROOT"
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"

# --------------------------------------------------------------------------
# No subject: run this script once per subject of the experiment
# --------------------------------------------------------------------------
if [ -z "${SUBJECT:-}" ]; then
    if [ -n "${SNN_CHECKPOINT_ROOT:-}${SPECK_CHECKPOINT_ROOT:-}${RESULTS_DIR:-}" ]; then
        echo "ERROR: SNN_CHECKPOINT_ROOT, SPECK_CHECKPOINT_ROOT and RESULTS_DIR are per subject; name a subject to set them" >&2
        exit 1
    fi
    SUBJECTS=()
    for d in $(find "${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}" -mindepth 2 -maxdepth 2 -type d \
                    -name per_session 2>/dev/null | sort); do
        SUBJECTS+=("$(basename "$(dirname "$d")")")
    done
    if [ "${#SUBJECTS[@]}" -eq 0 ]; then
        echo "ERROR: no ${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/<subject>/per_session/" >&2
        exit 1
    fi
    echo "Subjects of ${EXPERIMENT}: ${SUBJECTS[*]}"
    FAILED=()
    for subject in "${SUBJECTS[@]}"; do
        echo "##### ${EXPERIMENT}/${subject} #####"
        DATA_ROOT="$DATA_ROOT" bash "$0" "$RUN" "$EXPERIMENT" "$subject" || FAILED+=("$subject")
    done
    [ "${#FAILED[@]}" -gt 0 ] && { echo "Failed subjects (see above): ${FAILED[*]}"; exit 1; }
    exit 0
fi
if [ "${RUN:-}" = "--local" ]; then
    DECODERS="${DECODERS:-kf,wf,lstm,qrnn,snn,speck}"
    TRAIN_DURATIONS="${TRAIN_DURATIONS-}"
    MACHINE="${MACHINE:-laptop}"
    export ENERGY_METER_NUM_THREADS="${ENERGY_METER_NUM_THREADS:-1}"
fi

FEATURE="mua"
DECODERS="${DECODERS:-kf,wf,lstm,qrnn,snn}"
TEST_FRAC=0.1
TRAIN_DURATIONS="${TRAIN_DURATIONS-1,2,3,4,5,6,7,8,9,10}"
MACHINE="${MACHINE:-cluster}"

DATASET_ROOT="${DATA_ROOT}/dataset/${EXPERIMENT}/${SUBJECT}/mua"
MODEL_CACHE_ROOT="${DATA_ROOT}/results/model_cache/${EXPERIMENT}/${SUBJECT}"
# SNN checkpoints: ${SNN_CHECKPOINT_ROOT}/<session>/[${SNN_CHECKPOINT_SUBDIR}/]best_model_weights.pth
# (SNN_CHECKPOINT_SUBDIR selects one config of a sweep, e.g. wide768_2x)
SNN_CHECKPOINT_ROOT="${SNN_CHECKPOINT_ROOT:-${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/per_session}"
SNN_CHECKPOINT_SUBDIR="${SNN_CHECKPOINT_SUBDIR:-}"
# The 'speck' decoder runs the SNN checkpoints unless SPECK_CHECKPOINT_ROOT
# (and SPECK_CHECKPOINT_SUBDIR) name others, laid out the same way.
if [ -n "${SPECK_CHECKPOINT_ROOT:-}" ]; then
    SPECK_CHECKPOINT_SUBDIR="${SPECK_CHECKPOINT_SUBDIR:-}"
else
    SPECK_CHECKPOINT_ROOT="$SNN_CHECKPOINT_ROOT"
    SPECK_CHECKPOINT_SUBDIR="${SPECK_CHECKPOINT_SUBDIR:-$SNN_CHECKPOINT_SUBDIR}"
fi
if [ "$EXPERIMENT" = "hkm" ]; then
    SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"          # whole trials
else
    SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
fi
RESULTS_DIR="${RESULTS_DIR:-${DATA_ROOT}/results/test_all_decoders/${EXPERIMENT}/${SUBJECT}}"

show_settings() {
    echo "DATA_ROOT=${DATA_ROOT}"
    if [ "${MODE:-}" = "figures" ]; then
        echo "Figures (from saved predictions) for: ${FIGURE_DECODERS:-every decoder}"
    elif [ "${MODE:-}" != "report" ]; then
        echo "SNN checkpoints: ${SNN_CHECKPOINT_ROOT}${SNN_CHECKPOINT_SUBDIR:+ (/<session>/${SNN_CHECKPOINT_SUBDIR})}"
        [[ ",$DECODERS," == *,speck,* ]] && echo "Speck checkpoints: ${SPECK_CHECKPOINT_ROOT}${SPECK_CHECKPOINT_SUBDIR:+ (/<session>/${SPECK_CHECKPOINT_SUBDIR})}"
        echo "Decoders: ${DECODERS}${FIGURES:+; figures for ${FIGURE_DECODERS:-every decoder}}"
    fi
    echo "Results: ${RESULTS_DIR}"
}

snn_checkpoint() {  # snn_checkpoint SESSION_DIR
    echo "$1/${SNN_CHECKPOINT_SUBDIR:+${SNN_CHECKPOINT_SUBDIR}/}best_model_weights.pth"
}
speck_checkpoint() {  # speck_checkpoint SESSION
    echo "${SPECK_CHECKPOINT_ROOT}/$1/${SPECK_CHECKPOINT_SUBDIR:+${SPECK_CHECKPOINT_SUBDIR}/}best_model_weights.pth"
}

# Sessions = subdirectories of SNN_CHECKPOINT_ROOT holding a checkpoint.
SESSION_IDS=()
for d in $(find "$SNN_CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort); do
    [ -f "$(snn_checkpoint "$d")" ] && SESSION_IDS+=("$(basename "$d")")
done

# --------------------------------------------------------------------------
# SUBMIT: array + dependent report job
# --------------------------------------------------------------------------
if [ -n "${RUN:-}" ] && [ "${#SESSION_IDS[@]}" -eq 0 ]; then
    echo "ERROR: no $(snn_checkpoint '<session>') under $SNN_CHECKPOINT_ROOT" >&2
    exit 1
fi
# 'speck' is dropped from a session without its checkpoint, so say which ones.
if [ -n "${RUN:-}" ] && [[ ",$DECODERS," == *,speck,* ]]; then
    NO_SPECK=()
    for s in "${SESSION_IDS[@]}"; do
        [ -f "$(speck_checkpoint "$s")" ] || NO_SPECK+=("$s")
    done
    if [ "${#NO_SPECK[@]}" -eq "${#SESSION_IDS[@]}" ]; then
        echo "ERROR: 'speck' requested but no session has $(speck_checkpoint '<session>')" >&2
        echo "       (set SPECK_CHECKPOINT_ROOT / SPECK_CHECKPOINT_SUBDIR, or leave speck out of DECODERS)" >&2
        exit 1
    elif [ "${#NO_SPECK[@]}" -gt 0 ]; then
        echo "WARNING: no $(speck_checkpoint '<session>') for ${#NO_SPECK[@]} session(s), run without 'speck': ${NO_SPECK[*]}" >&2
    fi
fi
if [ "${RUN:-}" = "--submit" ]; then
    show_settings
    mkdir -p logs
    EXPORTS="ALL,EXPERIMENT=${EXPERIMENT},SUBJECT=${SUBJECT},DATA_ROOT=${DATA_ROOT},DECODERS=${DECODERS//,/:}"
    EXPORTS+=",TRAIN_DURATIONS=${TRAIN_DURATIONS//,/:},MACHINE=${MACHINE},SNN_CHECKPOINT_ROOT=${SNN_CHECKPOINT_ROOT}"
    EXPORTS+=",SNN_CHECKPOINT_SUBDIR=${SNN_CHECKPOINT_SUBDIR},SPECK_CHECKPOINT_ROOT=${SPECK_CHECKPOINT_ROOT}"
    EXPORTS+=",SPECK_CHECKPOINT_SUBDIR=${SPECK_CHECKPOINT_SUBDIR},RESULTS_DIR=${RESULTS_DIR},FIGURES=${FIGURES:-}"
    EXPORTS+=",FIGURE_DECODERS=${FIGURE_DECODERS//,/:},DURATIONS_JSON=${DURATIONS_JSON:-}"
    ARRAY_ID=$(sbatch --parsable --array=0-$(( ${#SESSION_IDS[@]} - 1 )) \
        --export="${EXPORTS},MODE=session" "$0")
    REPORT_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID%%;*}" --time=0:30:00 \
        -J INFERENCE_REPORT -o logs/INFERENCE_REPORT_%j.out -e logs/INFERENCE_REPORT_%j.out \
        --export="${EXPORTS},MODE=report" "$0")
    echo "Submitted ${#SESSION_IDS[@]} session task(s) as job $ARRAY_ID and the report as job $REPORT_ID"
    exit 0
fi

# Commas in exported values would split --export, so lists travel as a:b:c.
TRAIN_DURATIONS="${TRAIN_DURATIONS//:/,}"
DECODERS="${DECODERS//:/,}"
FIGURE_DECODERS="${FIGURE_DECODERS//:/,}"

VENV="${VENV:-$HOME/ann-env}"
if [ -f "$VENV/bin/activate" ]; then
    deactivate 2>/dev/null || true
    source "$VENV/bin/activate"
fi
export OMP_NUM_THREADS=${ENERGY_METER_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
# sbatch runs a copy of this script, so locate inference/ from the submit directory.
INFERENCE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/../inference"

run_report() {
    python3 "$INFERENCE_DIR/make_report.py" --results_dir "$RESULTS_DIR" \
        ${DURATIONS_JSON:+--durations_json "$DURATIONS_JSON"}
}

run_session() {  # run_session SESSION_ID
    local figure_args=()
    [ -n "${FIGURES:-}" ] && figure_args=(--figures_dir "${RESULTS_DIR}/figures/$1"
                                          --figure_decoders "${FIGURE_DECODERS:-}")
    python3 "$INFERENCE_DIR/test_all_decoders.py" \
        --experiment "$EXPERIMENT" \
        --session "$1" \
        --input_filepath "${DATASET_ROOT}/$1_binning.h5" \
        --model_dir "${MODEL_CACHE_ROOT}/$1" \
        --snn_checkpoint_path "$(snn_checkpoint "${SNN_CHECKPOINT_ROOT}/$1")" \
        --speck_checkpoint_path "$(speck_checkpoint "$1")" \
        --snn_dataset_path "${SNN_DATASET_ROOT}/$1" \
        --output "${RESULTS_DIR}/sessions/$1.json" \
        --feature "$FEATURE" \
        --decoders "$DECODERS" \
        --test_frac "$TEST_FRAC" \
        --train_durations "$TRAIN_DURATIONS" \
        --machine "$MACHINE" \
        ${SPECK_WAIT_TIME:+--speck_wait_time "$SPECK_WAIT_TIME"} \
        "${figure_args[@]}"
}

# --------------------------------------------------------------------------
# LOCAL: every session in series, then the report
# --------------------------------------------------------------------------
if [ "${RUN:-}" = "--local" ]; then
    show_settings
    FAILED=()
    for i in "${!SESSION_IDS[@]}"; do
        echo "=== Session ${SESSION_IDS[$i]} ($(( i + 1 ))/${#SESSION_IDS[@]}), host $(hostname) ==="
        run_session "${SESSION_IDS[$i]}" || FAILED+=("${SESSION_IDS[$i]}")
    done
    [ "${#FAILED[@]}" -gt 0 ] && echo "Failed sessions (see above): ${FAILED[*]}"
    run_report
    exit 0
fi

# --------------------------------------------------------------------------
# FIGURES: redraw every session's figures from its saved predictions
# --------------------------------------------------------------------------
if [ "${MODE:-}" = "figures" ]; then
    show_settings
    shopt -s nullglob
    ARRAYS=("${RESULTS_DIR}"/sessions/*_arrays.npz)
    if [ "${#ARRAYS[@]}" -eq 0 ]; then
        echo "ERROR: no saved predictions (sessions/<session>_arrays.npz) under ${RESULTS_DIR}" >&2
        exit 1
    fi
    FAILED=()
    for arrays in "${ARRAYS[@]}"; do
        session="$(basename "$arrays" _arrays.npz)"
        python3 "$INFERENCE_DIR/test_all_decoders.py" --figures_only \
            --experiment "$EXPERIMENT" --session "$session" \
            --input_filepath "${DATASET_ROOT}/${session}_binning.h5" \
            --model_dir "${MODEL_CACHE_ROOT}/${session}" \
            --output "${RESULTS_DIR}/sessions/${session}.json" \
            --figures_dir "${RESULTS_DIR}/figures/${session}" \
            --figure_decoders "${FIGURE_DECODERS:-}" || FAILED+=("$session")
    done
    [ "${#FAILED[@]}" -gt 0 ] && { echo "Failed sessions (see above): ${FAILED[*]}"; exit 1; }
    exit 0
fi

# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------
if [ "${MODE:-}" = "report" ]; then
    [ -z "${SLURM_JOB_ID:-}" ] && show_settings
    run_report
    exit 0
fi

# --------------------------------------------------------------------------
# SESSION (one array task)
# --------------------------------------------------------------------------
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#SESSION_IDS[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#SESSION_IDS[@]} found); exiting."
    exit 0
fi
echo "Session ${SESSION_IDS[${SLURM_ARRAY_TASK_ID:-0}]} ($(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))/${#SESSION_IDS[@]}), host $(hostname)"
run_session "${SESSION_IDS[${SLURM_ARRAY_TASK_ID:-0}]}"
