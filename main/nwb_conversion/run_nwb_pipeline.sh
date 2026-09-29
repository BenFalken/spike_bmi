#!/bin/bash
# Orchestrates the FULL NWB conversion pipeline for one session, start to
# finish.
#
#   1. convert_nwb_trials_to_raw_h5.py       NWB file -> one raw h5 per trial
#   2. run_dense_windowing_for_all_trials.sh (ANN) per-trial raw -> per-trial windowed
#   3. combine_trial_windows_to_ann_h5.py    (ANN)  -> {session}_{method}.h5
#   4. make_snn_dataset_whole_trial.py       (SNN, WHOLE-TRIAL) -> train/test/*.pkl
#
# THIS REVISION removes the old 3-stage windowed SNN path entirely
# (previously: run_windowing_for_all_trials.sh -> combine_trial_windows_to_
# session.py -> combine_trial_windows_to_grouped_session.py, producing
# mua_1_group/mua_8_group). Settled data-processing decision: HKM's own
# trials are short, independent reaches -- confirmed directly via
# compare_ann_snn_trial_coverage.py that a large fraction of real trials
# are shorter than even one 256ms SNN window, so windowing them was
# silently discarding real data rather than a neutral processing choice.
# make_snn_dataset_whole_trial.py replaces all three old stages at once:
# reads Stage 1's raw per-trial output DIRECTLY (no intermediate SNN
# windowing stage exists anymore at all -- there is nothing left to
# window), bins each trial's own COMPLETE duration in one pass, and
# writes whole-trial, variable-length train/test/*.pkl files straight to
# the final destination. Training on these needs --batch-size 1 (see
# that script's own module docstring for why dataset.py's existing
# collate_fn() already handles this with zero code changes).
#
# The OLD 3-stage path's own scripts (run_windowing_for_all_trials.sh,
# combine_trial_windows_to_session.py, combine_trial_windows_to_grouped_
# session.py) are DEAD as of this revision -- not called anywhere below.
# Left on disk for now rather than deleted outright, but should not be
# invoked directly; delete once you're confident nothing external still
# calls them.
#
# ANN/KF/WF processing (dense, near-total-overlap 4ms-step windows,
# bounded to individual trials -- confirmed already correct, unchanged
# behavior: run_dense_windowing_for_all_trials.sh runs make_dataset.py
# once per trial file, so dense windows structurally cannot cross a
# trial boundary) is untouched by this revision.
#
# DESTINATIONS: this revision also writes DIRECTLY to the real, final
# dataset roots (--dataset-root, --snn-dataset-root) instead of staging
# everything under --output-root and requiring a separate move step
# afterward -- --output-root now holds ONLY the genuinely intermediate,
# disposable per-trial files (raw + ANN-windowed), matching this
# project's own dataset/{experiment}/{subject}/mua/ and
# snn_datasets/{experiment}/{subject}/mua/ conventions used everywhere
# else (test_all_decoders.py, train_bmi.py, etc).
#
# EXPERIMENT is hardcoded to "hkm" here specifically -- this script only
# ever converts NWB data, which is always the hkm experiment (bmi's own
# subjects, indy/loco, come from the original .mat-based pipeline, not
# this one). SUBJECT is derived automatically from the NWB filename
# itself (sub-{Subject}_ses-... -> lowercased), rather than requiring a
# separate flag that could silently drift out of sync with the actual
# file being converted -- confirmed directly against both real filename
# patterns in this project's own data (sub-Nitschke_ses-...,
# sub-Jenkins_ses-...) before relying on it.
#
# INTERMEDIATE files (raw per-trial h5, ANN per-trial windowed h5) are
# disposable once the final outputs below exist; nothing downstream reads
# them. Pass --cleanup-intermediate to delete them automatically after a
# successful run. Left alone by default, since they're also the fastest
# way to re-run just stages 3-4 with a different --test-frac without
# repeating the (slower) conversion/windowing stages.
#
# Usage:
#   ./run_nwb_pipeline.sh --nwb-path PATH --output-root DIR \
#       --dataset-root DIR --snn-dataset-root DIR [options]
#
# Example:
#   ./run_nwb_pipeline.sh \
#       --nwb-path data/sub-Nitschke_ses-20100923_behavior+ecephys.nwb \
#       --output-root /users/bfalkenb/scratch/bfalkenb/data/hkm_intermediate \
#       --dataset-root /users/bfalkenb/scratch/bfalkenb/data/dataset \
#       --snn-dataset-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets \
#       --test-frac 0.1 --feature mua --method binning \
#       --cleanup-intermediate

set -eo pipefail

# --- Defaults ---
TEST_FRAC=0.1
MAX_GAP_MS=20.0
MIN_SAMPLES=5
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
        --feature) FEATURE="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --cleanup-intermediate) CLEANUP_INTERMEDIATE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$NWB_PATH" ] || [ -z "$OUTPUT_ROOT" ] || [ -z "$DATASET_ROOT" ] || [ -z "$SNN_DATASET_ROOT" ]; then
    echo "Usage: $0 --nwb-path PATH --output-root DIR --dataset-root DIR " \
         "--snn-dataset-root DIR [options]" >&2
    exit 1
fi

SESSION_ID="$(basename "$NWB_PATH" .nwb)"

# Derive SUBJECT from the NWB filename itself -- see header comment for
# why this is preferred over a separate, driftable --subject flag.
if [[ "$SESSION_ID" =~ sub-([A-Za-z]+)_ses- ]]; then
    SUBJECT="$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
else
    echo "ERROR: could not parse subject from filename '$SESSION_ID' -- expected the "
    echo "       'sub-{Subject}_ses-...' pattern (e.g. sub-Nitschke_ses-20090812_...)." >&2
    exit 1
fi
EXPERIMENT="hkm"

# --- Intermediate, disposable staging (under --output-root only) ---
RAW_DIR="${OUTPUT_ROOT}/raw_trials/${SESSION_ID}"
ANN_WINDOWED_DIR="${OUTPUT_ROOT}/ann_windowed_per_trial/${SESSION_ID}"

# --- Final, real destinations (under --dataset-root / --snn-dataset-root) ---
ANN_OUTPUT_DIR="${DATASET_ROOT}/${EXPERIMENT}/${SUBJECT}/mua"
ANN_OUTPUT_PATH="${ANN_OUTPUT_DIR}/${SESSION_ID}_${METHOD}.h5"
SNN_OUTPUT_ROOT="${SNN_DATASET_ROOT}"  # make_snn_dataset_whole_trial.py's own --dest-root;
# it appends {experiment}/{subject}/mua/{session_id}/{train,test} itself

echo "================================================================"
echo "NWB pipeline for session: $SESSION_ID"
echo "  experiment: $EXPERIMENT   subject: $SUBJECT"
echo "  nwb-path:    $NWB_PATH"
echo "  output-root (intermediate, disposable): $OUTPUT_ROOT"
echo "  dataset-root (ANN final):               $DATASET_ROOT"
echo "  snn-dataset-root (SNN final):            $SNN_DATASET_ROOT"
echo "================================================================"

echo ""
echo "--- Stage 1: NWB -> per-trial raw h5 ---"
python3 convert_nwb_trials_to_raw_h5.py \
    --nwb-path "$NWB_PATH" \
    --output-dir "$RAW_DIR" \
    --max-gap-ms "$MAX_GAP_MS" --min-samples "$MIN_SAMPLES"

echo ""
echo "--- Stage 2: per-trial raw -> per-trial windowed (ANN) ---"
./run_dense_windowing_for_all_trials.sh "$RAW_DIR" "$ANN_WINDOWED_DIR" "$METHOD"

echo ""
echo "--- Stage 3: combine -> {session}_${METHOD}.h5 (ANN/KF/WF), final destination ---"
mkdir -p "$ANN_OUTPUT_DIR"
python3 combine_trial_windows_to_ann_h5.py \
    --windowed-dir "$ANN_WINDOWED_DIR" \
    --output-path "$ANN_OUTPUT_PATH" \
    --test_frac "$TEST_FRAC"

echo ""
echo "--- Stage 4: whole-trial SNN dataset, final destination (no windowing/grouping) ---"
python3 make_snn_dataset_whole_trial.py \
    --output-root "$OUTPUT_ROOT" \
    --session-id "$SESSION_ID" \
    --dest-root "$SNN_OUTPUT_ROOT" \
    --experiment "$EXPERIMENT" --subject "$SUBJECT" \
    --feature "$FEATURE" --test_frac "$TEST_FRAC"

if [ "$CLEANUP_INTERMEDIATE" -eq 1 ]; then
    echo ""
    echo "--- Cleanup: removing intermediate directories ---"
    rm -rf "$RAW_DIR" "$ANN_WINDOWED_DIR"
    echo "Removed: $RAW_DIR, $ANN_WINDOWED_DIR"
fi

SNN_FINAL_DIR="${SNN_OUTPUT_ROOT}/${EXPERIMENT}/${SUBJECT}/mua/${SESSION_ID}"
echo ""
echo "================================================================"
echo "Done. Final outputs:"
echo "  ANN/KF/WF (--input_filepath): $ANN_OUTPUT_PATH"
echo "  SNN, whole-trial (--data-path): $SNN_FINAL_DIR"
echo "================================================================"
