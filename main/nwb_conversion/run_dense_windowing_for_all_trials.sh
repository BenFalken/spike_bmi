#!/bin/bash
# Run make_dataset.py (65-sample windows at 4 ms steps) once per raw trial
# file, so no ANN window ever spans two trials. Trials shorter than one
# window write nothing.
#
# Usage:
#   bash run_dense_windowing_for_all_trials.sh <raw_trials_dir> <windowed_output_dir> [method]

set -eo pipefail

RAW_DIR="$1"
OUT_DIR="$2"
METHOD="${3:-binning}"

if [ -z "$RAW_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: $0 <raw_trials_dir> <windowed_output_dir> [method]" >&2
    exit 1
fi
MAKE_DATASET="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../preprocessing_training/make_dataset.py"

mkdir -p "$OUT_DIR"

N=0
for raw_path in "$RAW_DIR"/*.h5; do
    stem="$(basename "$raw_path" .h5)"
    python3 "$MAKE_DATASET" \
        --input_filepath "$raw_path" \
        --output_filepath "${OUT_DIR}/${stem}_${METHOD}.h5" \
        --method "$METHOD" --wdw_time 0.256 --ol_time 0.252
    N=$((N + 1))
done

echo "Dense-windowed $N trial file(s) from $RAW_DIR into $OUT_DIR"
