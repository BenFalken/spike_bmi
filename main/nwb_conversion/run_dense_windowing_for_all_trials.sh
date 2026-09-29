#!/bin/bash
# ANN-side counterpart to run_windowing_for_all_trials.sh -- runs the
# REAL, unmodified make_dataset.py once per trial file, same reasoning:
# make_dataset.py's dense windowing walks task_time as one continuous
# array with no concept of a trial boundary, so it must run PER TRIAL,
# never on trial-concatenated raw data directly, or dense windows would
# straddle trial boundaries the same way make_snn_dataset.py's
# non-overlapping windows would have.
#
# Usage:
#   ./run_dense_windowing_for_all_trials.sh <raw_trials_dir> <windowed_output_dir> [method]
#
# Example:
#   ./run_dense_windowing_for_all_trials.sh raw_converted/nitschke_20100923 \
#       ann_windowed_per_trial/nitschke_20100923 binning

set -eo pipefail

RAW_DIR="$1"
OUT_DIR="$2"
METHOD="${3:-binning}"

if [ -z "$RAW_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: $0 <raw_trials_dir> <windowed_output_dir> [method]" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

N=0
for raw_path in "$RAW_DIR"/*.h5; do
    stem="$(basename "$raw_path" .h5)"
    out_path="${OUT_DIR}/${stem}_${METHOD}.h5"
    python3 ../preprocessing_training/make_dataset.py \
        --input_filepath "$raw_path" \
        --output_filepath "$out_path" \
        --method "$METHOD" --wdw_time 0.256 --ol_time 0.252
    N=$((N + 1))
done

echo "Dense-windowed $N trial file(s) from $RAW_DIR into $OUT_DIR"
