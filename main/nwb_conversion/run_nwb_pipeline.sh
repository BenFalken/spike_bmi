#!/bin/bash
# Build the ANN and SNN datasets of one HKM (NWB) session.
#
#   1. convert_nwb_trials_to_raw_h5.py        NWB -> one raw .h5 per trial, each from its own
#                                             recording run (nwb_pieces.py), tracking glitches
#                                             removed (--max-speed)
#   2. run_dense_windowing_for_all_trials.sh  make_dataset.py on each trial (65-sample
#                                             windows at 4 ms steps, never across trials)
#   3. combine_trial_windows_to_ann_h5.py     -> ${DATASET_ROOT}/hkm/<subject>/<feature>/<session>_<method>.h5
#   4. make_snn_dataset_whole_trial.py        -> ${SNN_DATASET_ROOT}/hkm/<subject>/<feature>/<session>/{train,test}/*.pkl
#                                             (one whole trial per .pkl; fails if any glitch survived)
#
# Stages 3 and 4 share one whole-trial train/test split (trial_split.py), so
# both datasets hold out the same trials. The subject is read from the NWB
# file name (sub-<Subject>_ses-...).
#
# Every run rebuilds the session from scratch: it deletes the session's
# intermediate files (under --output-root) and previous final outputs first,
# so nothing from an earlier run can mix in. Sessions never share files, so
# any number can run at once. --cleanup-intermediate deletes the per-trial
# intermediates after a successful run (the conversion report is kept under
# ${OUTPUT_ROOT}/conversion_reports/).
#
# The pipeline's Python scripts are always the ones next to this file,
# whatever directory it is run from.
#
# Usage:
#   bash run_nwb_pipeline.sh --nwb-path PATH --output-root DIR \
#       --dataset-root DIR --snn-dataset-root DIR [--test-frac 0.1] [--max-gap-ms 20]
#       [--min-samples 5] [--max-speed 3000] [--feature mua] [--method binning]
#       [--cleanup-intermediate]
#
# Example:
#   bash run_nwb_pipeline.sh \
#       --nwb-path $DATA_ROOT/raw/hkm/jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb \
#       --output-root $DATA_ROOT/hkm_intermediate \
#       --dataset-root $DATA_ROOT/dataset --snn-dataset-root $DATA_ROOT/snn_datasets

set -eo pipefail

TEST_FRAC=0.1
MAX_GAP_MS=20.0
MIN_SAMPLES=5
MAX_SPEED=3000
FEATURE=mua
METHOD=binning
CLEANUP_INTERMEDIATE=0
NWB_PATH=""
OUTPUT_ROOT=""
DATASET_ROOT=""
SNN_DATASET_ROOT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --nwb-path) NWB_PATH="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --snn-dataset-root) SNN_DATASET_ROOT="$2"; shift 2 ;;
        --test-frac) TEST_FRAC="$2"; shift 2 ;;
        --max-gap-ms) MAX_GAP_MS="$2"; shift 2 ;;
        --min-samples) MIN_SAMPLES="$2"; shift 2 ;;
        --max-speed) MAX_SPEED="$2"; shift 2 ;;
        --feature) FEATURE="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --cleanup-intermediate) CLEANUP_INTERMEDIATE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$NWB_PATH" ] || [ -z "$OUTPUT_ROOT" ] || [ -z "$DATASET_ROOT" ] || [ -z "$SNN_DATASET_ROOT" ]; then
    echo "Usage: $0 --nwb-path PATH --output-root DIR --dataset-root DIR --snn-dataset-root DIR [options]" >&2
    exit 1
fi

# Make the paths absolute before moving to this script's directory.
NWB_PATH="$(realpath -m "$NWB_PATH")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
DATASET_ROOT="$(realpath -m "$DATASET_ROOT")"
SNN_DATASET_ROOT="$(realpath -m "$SNN_DATASET_ROOT")"
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

SESSION_ID="$(basename "$NWB_PATH" .nwb)"
if [[ "$SESSION_ID" =~ ^sub-([A-Za-z]+)_ses- ]]; then
    SUBJECT="$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
else
    echo "ERROR: cannot read the subject from '$SESSION_ID' (expected sub-<Subject>_ses-...)" >&2
    exit 1
fi
EXPERIMENT="hkm"

RAW_DIR="${OUTPUT_ROOT}/raw_trials/${SESSION_ID}"
ANN_WINDOWED_DIR="${OUTPUT_ROOT}/ann_windowed_per_trial/${SESSION_ID}"
REPORT_DIR="${OUTPUT_ROOT}/conversion_reports"
ANN_OUTPUT_PATH="${DATASET_ROOT}/${EXPERIMENT}/${SUBJECT}/${FEATURE}/${SESSION_ID}_${METHOD}.h5"
SNN_OUTPUT_DIR="${SNN_DATASET_ROOT}/${EXPERIMENT}/${SUBJECT}/${FEATURE}/${SESSION_ID}"

echo "================================================================"
echo "NWB pipeline: $SESSION_ID ($EXPERIMENT/$SUBJECT)"
echo "  nwb:           $NWB_PATH"
echo "  intermediate:  $OUTPUT_ROOT"
echo "  ANN dataset:   $ANN_OUTPUT_PATH"
echo "  SNN dataset:   $SNN_OUTPUT_DIR"
echo "  test-frac=$TEST_FRAC max-gap-ms=$MAX_GAP_MS min-samples=$MIN_SAMPLES max-speed=$MAX_SPEED"
echo "  code:          $(pwd)"
echo "================================================================"

# Start clean: nothing from an earlier run may survive into this one.
rm -rf "$RAW_DIR" "$ANN_WINDOWED_DIR" "${SNN_OUTPUT_DIR}/train" "${SNN_OUTPUT_DIR}/test"
rm -f "$ANN_OUTPUT_PATH"
mkdir -p "$REPORT_DIR" "$(dirname "$ANN_OUTPUT_PATH")"

echo ""
echo "--- Stage 1: NWB -> per-trial raw h5 ---"
python3 convert_nwb_trials_to_raw_h5.py \
    --nwb-path "$NWB_PATH" --output-dir "$RAW_DIR" \
    --max-gap-ms "$MAX_GAP_MS" --min-samples "$MIN_SAMPLES" --max-speed "$MAX_SPEED" --overwrite
cp -f "${RAW_DIR}/${SESSION_ID}_conversion_report.json" "$REPORT_DIR/"

echo ""
echo "--- Stage 2: per-trial raw -> per-trial dense windows (ANN) ---"
bash run_dense_windowing_for_all_trials.sh "$RAW_DIR" "$ANN_WINDOWED_DIR" "$METHOD"

echo ""
echo "--- Stage 3: ANN dataset ---"
python3 combine_trial_windows_to_ann_h5.py \
    --raw-dir "$RAW_DIR" --windowed-dir "$ANN_WINDOWED_DIR" \
    --output-path "$ANN_OUTPUT_PATH" --method "$METHOD" --test_frac "$TEST_FRAC"

echo ""
echo "--- Stage 4: whole-trial SNN dataset ---"
python3 make_snn_dataset_whole_trial.py \
    --raw-dir "$RAW_DIR" --session-id "$SESSION_ID" --dest-root "$SNN_DATASET_ROOT" \
    --experiment "$EXPERIMENT" --subject "$SUBJECT" --feature "$FEATURE" \
    --test_frac "$TEST_FRAC" --max-speed "$MAX_SPEED"

if [ "$CLEANUP_INTERMEDIATE" -eq 1 ]; then
    rm -rf "$RAW_DIR" "$ANN_WINDOWED_DIR"
    echo "Removed intermediates: $RAW_DIR, $ANN_WINDOWED_DIR"
fi

echo ""
echo "================================================================"
echo "Done: $SESSION_ID"
echo "  ANN dataset (KF/WF/LSTM/QRNN): $ANN_OUTPUT_PATH"
echo "  SNN dataset (whole trials):    $SNN_OUTPUT_DIR"
echo "================================================================"
