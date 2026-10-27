#!/bin/bash
# Multi-session version of compare_lsnn.py, for a laptop: rebuilds each session's SNN dataset from its raw .mat
# (process_data -> make_snn_dataset -> combine_snn_dataset, the pipeline's settings) when missing, then trains and
# compares, per session, the LSNN (2 readout stages) against the LOSO model's network trained on that session alone
# (pooled_soft) and the same with unit-gain synaptic filters (pooled_soft_unitgain). Finished models are skipped.
#   RAW_DIR=~/Downloads ROOT=~/Downloads/spike_bmi_lsnn_data bash run_sessions_local.sh indy_20160411_01 indy_20160921_01 ...
# LSNN_NAME / LSNN_ARGS pick the LSNN variant (default lsnn_out2: --lsnn-out-stages 2).
# Summary over sessions: $ROOT/lsnn_vs_snn/sessions_summary_$LSNN_NAME.md
set -eo pipefail
RAW_DIR="${RAW_DIR:-$HOME/Downloads}"
ROOT="${ROOT:-$HOME/Downloads/spike_bmi_lsnn_data}"
DEVICE="${DEVICE:-cpu}"
LSNN_NAME="${LSNN_NAME:-lsnn_out2}"                       # the LSNN variant: its directory name and train_snn flags
LSNN_ARGS="${LSNN_ARGS:---lsnn-out-stages 2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PRE="$HERE/../preprocessing_training"
for S in "$@"; do
    SUBJ="${S%%_*}"
    DATA="$ROOT/snn_datasets/bmi/$SUBJ/mua_8_group/$S"
    if [ ! -f "$DATA/train/0.pkl" ]; then
        echo "=== $S: building the dataset from $RAW_DIR/$S.mat"
        mkdir -p "$ROOT/processed/bmi/$SUBJ/mua" "$ROOT/dataset/bmi/$SUBJ/mua"
        python3 "$PRE/process_data.py" --input_filepath "$RAW_DIR/$S.mat" \
            --output_filepath "$ROOT/processed/bmi/$SUBJ/mua/$S.h5"
        python3 "$PRE/make_snn_dataset.py" --input_filepath "$ROOT/processed/bmi/$SUBJ/mua/$S.h5" \
            --output_filepath "$ROOT/dataset/bmi/$SUBJ/mua/${S}_snn.h5" --feature mua --wdw_time 0.256 --ol_time 0
        python3 "$PRE/combine_snn_dataset.py" --h5-path "$ROOT/dataset/bmi/$SUBJ/mua/${S}_snn.h5" --session-id "$S" \
            --group-size 8 --test-frac 0.1 --output-dir "$ROOT/snn_datasets/bmi/$SUBJ/mua_8_group" --discard-train-remainder
    fi
    OUT="$ROOT/lsnn_vs_snn/$S"
    mkdir -p "$OUT"
    for CFG in pooled_soft pooled_soft_unitgain; do
        echo "=== $S: $CFG vs lsnn_out2"
        python3 "$HERE/compare_lsnn.py" --data-path "$DATA" --out-dir "$OUT" --device "$DEVICE" --snn-config "$CFG" \
            --lsnn-name "$LSNN_NAME" --lsnn-args $LSNN_ARGS > "$OUT/run_${CFG}_${LSNN_NAME}.log" 2>&1
    done
done
python3 - "$ROOT/lsnn_vs_snn" "$LSNN_NAME" "$@" <<'EOF'
import json, os, sys
import numpy as np
root, lsnn, sessions = sys.argv[1], sys.argv[2], sys.argv[3:]
cols = [("snn_pooled_soft", "pooled_soft"), ("snn_pooled_soft_unitgain", "pooled_soft_unitgain"), (lsnn, "pooled_soft")]
labels = ["SNN (LOSO network, alone)", "SNN, unit-gain filters", "LSNN"]
rows = {}
for s in sessions:
    rows[s] = {}
    for key, cfg in cols:
        f = os.path.join(root, s, f"comparison_{cfg}_{lsnn}.json")
        if os.path.isfile(f):
            rows[s][key] = json.load(open(f))[key]
lines = ["# SNN vs LSNN over sessions (each trained on its session alone, seed 42)", "",
         "| session | " + " | ".join(f"{l}: error" for l in labels) + " | " + " | ".join(f"{l}: R²" for l in labels) + " |",
         "|---|" + "---|" * 6]
for s in sessions:
    r = rows[s]
    lines.append(f"| {s} | " + " | ".join(f"{r[k]['rmse']:.1f}" if k in r else "-" for k, _ in cols) + " | "
                 + " | ".join(f"{r[k]['r2_mean']:.3f}" if k in r else "-" for k, _ in cols) + " |")
done = [s for s in sessions if all(k in rows[s] for k, _ in cols)]
if done:
    lines.append(f"| **mean ({len(done)} sessions)** | " + " | ".join(
        f"**{np.mean([rows[s][k]['rmse'] for s in done]):.1f}**" for k, _ in cols) + " | " + " | ".join(
        f"**{np.mean([rows[s][k]['r2_mean'] for s in done]):.3f}**" for k, _ in cols) + " |")
    for k, l in zip([c[0] for c in cols[:2]], labels[:2]):
        wins = sum(rows[s][lsnn]["rmse"] < rows[s][k]["rmse"] for s in done)
        lines.append(f"\nLSNN lower error than {l}: {wins} of {len(done)} sessions.")
open(os.path.join(root, f"sessions_summary_{lsnn}.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
