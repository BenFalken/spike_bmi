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
# always rebuilt. Settings below can be overridden from the environment, e.g.
#   SNN_CHECKPOINT_ROOT=.../full_cohort_finetuned_medium bash run_inference.sbatch --submit bmi indy
#   FIGURES=1 bash run_inference.sbatch --submit bmi indy     # also per-session diagnostic figures
#   DATA_ROOT=~/data VENV=~/speck-env bash run_inference.sbatch --local bmi indy
#   (--local without DATA_ROOT uses the data/ folder next to main/, if there is one)
#   SPECK_WAIT_TIME=0.005 ...   seconds the chip gets per timestep before its output is read
#   Only rebuild the report:  MODE=report EXPERIMENT=bmi SUBJECT=indy bash run_inference.sbatch

set -eo pipefail

case "${1:-}" in
    --submit|--local) RUN="$1"; EXPERIMENT="${2:-}"; SUBJECT="${3:-}" ;;
esac
if [ -z "${EXPERIMENT:-}" ] || { [ -z "${SUBJECT:-}" ] && [ -z "${RUN:-}" ]; }; then
    echo "Usage: bash $0 --submit|--local <experiment> [<subject>]" >&2
    exit 1
fi
# Data root: $DATA_ROOT if exported; for --local, else the data/ folder next to
# main/ (repo/main/../data) when it exists; else the cluster scratch directory.
LOCAL_DATA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/data"
if [ -z "${DATA_ROOT:-}" ] && [ "${RUN:-}" = "--local" ] && [ -d "$LOCAL_DATA_ROOT" ]; then
    DATA_ROOT="$LOCAL_DATA_ROOT"
fi
DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"

# --------------------------------------------------------------------------
# No subject: run this script once per subject of the experiment
# --------------------------------------------------------------------------
if [ -z "${SUBJECT:-}" ]; then
    if [ -n "${SNN_CHECKPOINT_ROOT:-}" ] || [ -n "${RESULTS_DIR:-}" ]; then
        echo "ERROR: SNN_CHECKPOINT_ROOT and RESULTS_DIR are per subject; name a subject to set them" >&2
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
if [ "$EXPERIMENT" = "hkm" ]; then
    SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"          # whole trials
else
    SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
fi
RESULTS_DIR="${RESULTS_DIR:-${DATA_ROOT}/results/test_all_decoders/${EXPERIMENT}/${SUBJECT}}"

snn_checkpoint() {  # snn_checkpoint SESSION_DIR
    echo "$1/${SNN_CHECKPOINT_SUBDIR:+${SNN_CHECKPOINT_SUBDIR}/}best_model_weights.pth"
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
if [ "${RUN:-}" = "--submit" ]; then
    mkdir -p logs
    EXPORTS="ALL,EXPERIMENT=${EXPERIMENT},SUBJECT=${SUBJECT},DATA_ROOT=${DATA_ROOT},DECODERS=${DECODERS//,/:}"
    EXPORTS+=",TRAIN_DURATIONS=${TRAIN_DURATIONS//,/:},MACHINE=${MACHINE},SNN_CHECKPOINT_ROOT=${SNN_CHECKPOINT_ROOT}"
    EXPORTS+=",SNN_CHECKPOINT_SUBDIR=${SNN_CHECKPOINT_SUBDIR},RESULTS_DIR=${RESULTS_DIR},FIGURES=${FIGURES:-}"
    ARRAY_ID=$(sbatch --parsable --array=0-$(( ${#SESSION_IDS[@]} - 1 )) \
        --export="${EXPORTS},MODE=session" "$0")
    REPORT_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID%%;*}" --time=0:30:00 \
        -J INFERENCE_REPORT -o logs/INFERENCE_REPORT_%j.out -e logs/INFERENCE_REPORT_%j.out \
        --export="${EXPORTS},MODE=report" "$0")
    echo "Submitted ${#SESSION_IDS[@]} session task(s) as job $ARRAY_ID and the report as job $REPORT_ID"
    echo "Results: $RESULTS_DIR"
    exit 0
fi

# Commas in exported values would split --export, so lists travel as a:b:c.
TRAIN_DURATIONS="${TRAIN_DURATIONS//:/,}"
DECODERS="${DECODERS//:/,}"

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
    python3 "$INFERENCE_DIR/make_report.py" --results_dir "$RESULTS_DIR"
}

run_session() {  # run_session SESSION_ID
    local figure_args=()
    [ -n "${FIGURES:-}" ] && figure_args=(--figures_dir "${RESULTS_DIR}/figures/$1")
    python3 "$INFERENCE_DIR/test_all_decoders.py" \
        --experiment "$EXPERIMENT" \
        --session "$1" \
        --input_filepath "${DATASET_ROOT}/$1_binning.h5" \
        --model_dir "${MODEL_CACHE_ROOT}/$1" \
        --snn_checkpoint_path "$(snn_checkpoint "${SNN_CHECKPOINT_ROOT}/$1")" \
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
    echo "DATA_ROOT=${DATA_ROOT}"
    echo "SNN checkpoints: ${SNN_CHECKPOINT_ROOT}${SNN_CHECKPOINT_SUBDIR:+ (/<session>/${SNN_CHECKPOINT_SUBDIR})}"
    echo "Results: ${RESULTS_DIR}"
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
# REPORT
# --------------------------------------------------------------------------
if [ "${MODE:-}" = "report" ]; then
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
