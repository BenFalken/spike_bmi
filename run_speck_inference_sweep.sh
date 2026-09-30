#!/bin/bash
#
# Runs infer_snn_speck.py once per session, STRICTLY SEQUENTIALLY (same
# reasoning as before: the physical Speck2f devkit is a single, shared
# resource, never run in parallel), for every session belonging to one
# (DATASET, SUBJECT) pair -- e.g. DATASET=bmi SUBJECT=indy, or
# DATASET=hkm SUBJECT=jenkins.
#
# Path structure confirmed directly against a real, working command
# rather than assumed -- and it's ASYMMETRIC, so this matches that
# asymmetry exactly rather than a simpler, guessed structure:
#   checkpoints: {dataset}/mua/{subject}/{session}/best_model_weights.pth
#                (subject IS its own directory level)
#   datasets:    {dataset}/mua/{session}
#                (NO subject directory level -- session_id's own
#                {subject}_{date}_{index} prefix is what disambiguates)
#
# EDIT the four *_ROOT paths below if your actual layout differs,
# especially for hkm -- only the bmi/indy structure above has been
# directly confirmed; hkm is assumed consistent with it, not verified.
#
# Run this ON YOUR MAC (same machine infer_snn_speck.py itself runs on).
#
#   chmod +x run_speck_inference_sweep.sh
#   ./run_speck_inference_sweep.sh bmi indy
#   ./run_speck_inference_sweep.sh hkm jenkins

set -uo pipefail
# NOTE: -e deliberately NOT set -- see per-session error handling below,
# which needs one bad session to not abort the whole sweep.

DATASET="${1:-}"
SUBJECT="${2:-}"
if [ -z "$DATASET" ] || [ -z "$SUBJECT" ]; then
    echo "Usage: $0 <dataset: bmi|hkm> <subject: e.g. indy, loco, jenkins, nitschke>" >&2
    exit 1
fi

# --- Paths -- EDIT if your actual layout differs ---
CHECKPOINTS_ROOT="./checkpoints/${DATASET}/mua/${SUBJECT}"
DATASETS_ROOT="./datasets/${DATASET}/mua"
OUTPUT_ROOT="./speck_results/${DATASET}/${SUBJECT}"
INFER_SCRIPT="./infer_snn_speck.py"

# --- Which implementations to run ---
MODELS="torch speck"

if [ ! -f "$INFER_SCRIPT" ]; then
    echo "ERROR: $INFER_SCRIPT not found -- run this from the directory " >&2
    echo "       containing infer_snn_speck.py, or edit INFER_SCRIPT above." >&2
    exit 1
fi
if [ ! -d "$CHECKPOINTS_ROOT" ]; then
    echo "ERROR: CHECKPOINTS_ROOT does not exist: $CHECKPOINTS_ROOT" >&2
    exit 1
fi

# {session_id}/ directly under CHECKPOINTS_ROOT -- flat, matching the
# real, confirmed bmi/indy structure (no further nesting).
mapfile -t SESSION_DIRS < <(find "$CHECKPOINTS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
N_SESSIONS=${#SESSION_DIRS[@]}
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: found zero session directories under $CHECKPOINTS_ROOT" >&2
    exit 1
fi

echo "Found $N_SESSIONS session(s) under $CHECKPOINTS_ROOT"
echo "Running sequentially (chip does not support parallel access) -- models: $MODELS"
echo ""

FAILED_SESSIONS=()
SKIPPED_COUNT=0
RUN_COUNT=0

for i in "${!SESSION_DIRS[@]}"; do
    SESSION_DIR="${SESSION_DIRS[$i]}"
    SESSION_ID="$(basename "$SESSION_DIR")"

    CHECKPOINT_PATH="${SESSION_DIR}/best_model_weights.pth"
    DATASET_PATH="${DATASETS_ROOT}/${SESSION_ID}"
    SESSION_OUTPUT_DIR="${OUTPUT_ROOT}/${SESSION_ID}"

    echo "=== [$((i+1))/${N_SESSIONS}] ${SESSION_ID} ==="

    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: checkpoint not found at $CHECKPOINT_PATH -- skipping"
        FAILED_SESSIONS+=("$SESSION_ID (missing checkpoint)")
        continue
    fi
    if [ ! -d "$DATASET_PATH" ]; then
        echo "  ERROR: dataset not found at $DATASET_PATH -- skipping"
        FAILED_SESSIONS+=("$SESSION_ID (missing dataset)")
        continue
    fi

    # BUG FIX: this used to check for metrics.json, but infer_snn_speck.py was changed
    # to write speck_results.json instead (renamed when CC + param count were added, and
    # aggregate_speck_results.py reads that name). The old check could therefore never
    # match, so re-running this script silently redid EVERY session from scratch -- on a
    # single shared physical chip, that's the expensive way to be wrong. A leftover
    # metrics.json from an older run deliberately does NOT count as "done": those runs
    # predate the CC/param-count fields the aggregator now expects.
    if [ -f "${SESSION_OUTPUT_DIR}/speck_results.json" ]; then
        echo "  [skip] ${SESSION_OUTPUT_DIR}/speck_results.json already exists -- delete it "
        echo "         first if you actually want to redo this session."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    mkdir -p "$SESSION_OUTPUT_DIR"
    # --experiment/--subject passed explicitly: DATASETS_ROOT has no subject directory
    # level (see the path note in this file's header), so dataset.py can't derive the
    # subject from DATASET_PATH itself -- it needs to be told. --experiment also selects
    # which model module (model_bmi vs model_hkm) the checkpoint loads through, which
    # previously silently defaulted to bmi even for `./run_speck_inference_sweep.sh hkm ...`.
    python3 "$INFER_SCRIPT" \
        --experiment "$DATASET" \
        --subject "$SUBJECT" \
        --checkpoint-path "$CHECKPOINT_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$SESSION_OUTPUT_DIR" \
        --models $MODELS
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "  ERROR: ${SESSION_ID} exited with code ${exit_code} -- continuing to next session"
        FAILED_SESSIONS+=("$SESSION_ID (exit code ${exit_code})")
    else
        RUN_COUNT=$((RUN_COUNT + 1))
        echo "  OK: ${SESSION_ID} complete -> ${SESSION_OUTPUT_DIR}"
    fi
    echo ""
done

echo "======================================================================"
echo "Sweep complete for ${DATASET}/${SUBJECT}: ${RUN_COUNT} run, ${SKIPPED_COUNT} already-done skipped, "
echo "${#FAILED_SESSIONS[@]} failed, out of ${N_SESSIONS} total"
if [ "${#FAILED_SESSIONS[@]}" -gt 0 ]; then
    echo ""
    echo "Failed sessions:"
    for f in "${FAILED_SESSIONS[@]}"; do
        echo "  - $f"
    done
fi
