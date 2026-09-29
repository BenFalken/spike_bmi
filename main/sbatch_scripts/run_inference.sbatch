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
#   bash run_inference.sbatch --submit bmi indy
#
# submits (from this directory):
#   1. an array job, one task per session with an SNN checkpoint, running
#      inference/test_all_decoders.py: accuracy (full data and per training
#      duration) and latency / energy profiling -> sessions/<session>.json
#   2. a report job that starts when the array has finished, running
#      inference/make_report.py: combined_metrics*.json, efficiency_summary.json
#      and the efficiency, energy and 4x2 figures
# all under ${DATA_ROOT}/results/test_all_decoders/<experiment>/<subject>/.
#
# Re-submitting is safe: finished session sections are skipped, and the
# report is always rebuilt. Put the Speck laptop summary (aggregate_speck_results.py
# output) at <results dir>/aggregated_summary.json to include it in the figures.
#
# Settings below can be overridden from the environment when submitting, e.g.
#   SNN_CHECKPOINT_ROOT=.../full_cohort_finetuned_medium bash run_inference.sbatch --submit bmi indy
#   FIGURES=1 bash run_inference.sbatch --submit bmi indy     # also per-session diagnostic figures
#   Only rebuild the report:  MODE=report EXPERIMENT=bmi SUBJECT=indy bash run_inference.sbatch

set -eo pipefail

if [ "${1:-}" = "--submit" ]; then
    EXPERIMENT="$2"; SUBJECT="$3"
fi
if [ -z "${EXPERIMENT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "Usage: bash $0 --submit <experiment> <subject>" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-/users/bfalkenb/scratch/bfalkenb/data}"
FEATURE="mua"
DECODERS="${DECODERS:-kf,wf,lstm,qrnn,snn}"
TEST_FRAC=0.1
TRAIN_DURATIONS="${TRAIN_DURATIONS:-1,2,3,4,5,6,7,8,9,10}"

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
if [ "${1:-}" = "--submit" ]; then
    if [ "${#SESSION_IDS[@]}" -eq 0 ]; then
        echo "ERROR: no $(snn_checkpoint '<session>') under $SNN_CHECKPOINT_ROOT" >&2
        exit 1
    fi
    mkdir -p logs
    EXPORTS="ALL,EXPERIMENT=${EXPERIMENT},SUBJECT=${SUBJECT},DATA_ROOT=${DATA_ROOT},DECODERS=${DECODERS//,/:}"
    EXPORTS+=",TRAIN_DURATIONS=${TRAIN_DURATIONS//,/:},SNN_CHECKPOINT_ROOT=${SNN_CHECKPOINT_ROOT}"
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

deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------
if [ "${MODE:-}" = "report" ]; then
    python3 ../inference/make_report.py --results_dir "$RESULTS_DIR"
    exit 0
fi

# --------------------------------------------------------------------------
# SESSION (one array task)
# --------------------------------------------------------------------------
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#SESSION_IDS[@]}" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#SESSION_IDS[@]} found); exiting."
    exit 0
fi
SESSION_ID="${SESSION_IDS[${SLURM_ARRAY_TASK_ID:-0}]}"
FIGURE_ARGS=()
[ -n "${FIGURES:-}" ] && FIGURE_ARGS=(--figures_dir "${RESULTS_DIR}/figures/${SESSION_ID}")

echo "Session ${SESSION_ID} ($(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))/${#SESSION_IDS[@]}), host $(hostname)"
python3 ../inference/test_all_decoders.py \
    --experiment "$EXPERIMENT" \
    --session "$SESSION_ID" \
    --input_filepath "${DATASET_ROOT}/${SESSION_ID}_binning.h5" \
    --model_dir "${MODEL_CACHE_ROOT}/${SESSION_ID}" \
    --snn_checkpoint_path "$(snn_checkpoint "${SNN_CHECKPOINT_ROOT}/${SESSION_ID}")" \
    --snn_dataset_path "${SNN_DATASET_ROOT}/${SESSION_ID}" \
    --output "${RESULTS_DIR}/sessions/${SESSION_ID}.json" \
    --feature "$FEATURE" \
    --decoders "$DECODERS" \
    --test_frac "$TEST_FRAC" \
    --train_durations "$TRAIN_DURATIONS" \
    "${FIGURE_ARGS[@]}"
