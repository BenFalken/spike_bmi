#!/bin/bash
# Everything in sequence on one machine (the Speck laptop): validate the
# emulator against the study's chip run, then train every regime one after
# another, score each as PyTorch, quantized, emulated chip and the real chip,
# and write the summary table and figures.
#
#   cd main/speck_lab
#   DATA=~/.../data SESSION=sub-Jenkins_ses-20090912_behavior+ecephys bash run_lab_local.sh
#
# Settings (environment):
#   DATA        data root holding snn_datasets/ and snn_checkpoints/ (required; an
#               absolute path, or relative to this folder)
#   SESSION     session directory name (required)
#   SUBJECT     hkm subject (default jenkins)
#   CHECKPOINT  checkpoint the study ran, for the validation (default:
#               $DATA/snn_checkpoints/hkm/$SUBJECT/per_session_binary/$SESSION/best_model_weights.pth)
#   STUDY_JSON  the study's result (default ../../docs/data/speck_activity_jenkins_20090912.json)
#   REGIMES     regimes.py names, in order (default below)
#   OUT_DIR     results (default lab_runs/$SUBJECT)
#   DEVKIT      chip to score on (default speck2fdevkit:0; "none" for host only)
#   N_TRAIN, N_VAL, N_TEST, EPOCHS   trial counts and epochs (lab.py run defaults)
#   SHUFFLE     emulator event order: none, input or all (default none; pick the
#               one validation shows closest to the chip)
#   INIT        fine-tune every regime from this checkpoint (weights and velocity scale)
#               instead of training from scratch; use a separate OUT_DIR
#   SKIP_VALIDATE=1  skip step 1
#
# Re-running resumes: a regime whose model is already in OUT_DIR is only re-scored.
# Everything printed also goes to $OUT_DIR/run_lab_local.log.

set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # this folder (portable: no realpath)

: "${DATA:?set DATA to the data root (holding snn_datasets/ and snn_checkpoints/)}"
: "${SESSION:?set SESSION to the session directory name}"
SUBJECT="${SUBJECT:-jenkins}"
SNN_DATASET="${DATA}/snn_datasets/hkm/${SUBJECT}/mua/${SESSION}"
CHECKPOINT="${CHECKPOINT:-${DATA}/snn_checkpoints/hkm/${SUBJECT}/per_session_binary/${SESSION}/best_model_weights.pth}"
STUDY_JSON="${STUDY_JSON:-../../docs/data/speck_activity_jenkins_20090912.json}"
REGIMES="${REGIMES:-baseline rate10 rate30 rate10_sparse rate10_single rate10_emusel}"
OUT_DIR="${OUT_DIR:-lab_runs/${SUBJECT}}"
DEVKIT="${DEVKIT:-speck2fdevkit:0}"
SHUFFLE="${SHUFFLE:-none}"

for split in train test; do
    if [ -z "$(ls -A "${SNN_DATASET}/${split}" 2>/dev/null)" ]; then
        echo "ERROR: no ${SNN_DATASET}/${split}/*.pkl (training needs the session's train split too)" >&2
        exit 1
    fi
done
if [ -n "${INIT:-}" ] && [ ! -f "$INIT" ]; then
    echo "ERROR: INIT=${INIT} does not exist" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/run_lab_local.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T')  ${SNN_DATASET} -> ${OUT_DIR}"
echo "    regimes: ${REGIMES} | chip: ${DEVKIT} | emulator order: ${SHUFFLE}"
echo "    start: ${INIT:-from scratch}"

# --- 1. emulator against the real chip (host only, about a minute) ---
if [ "${SKIP_VALIDATE:-0}" != "1" ]; then
    if [ -f "$CHECKPOINT" ] && [ -f "$STUDY_JSON" ]; then
        echo -e "\n=== 1/3 validate the emulator"
        python3 lab.py validate --checkpoint "$CHECKPOINT" --snn_dataset "$SNN_DATASET" \
            --study_json "$STUDY_JSON" --output "${OUT_DIR}/validation.json" \
            || echo "WARNING: validation failed (see above); continuing with training"
    else
        echo -e "\n=== 1/3 skipped: no ${CHECKPOINT} or ${STUDY_JSON}"
    fi
fi

# --- 2. train and score every regime, one after another ---
echo -e "\n=== 2/3 train and score: ${REGIMES}"
RUN_ARGS=(--snn_dataset "$SNN_DATASET" --out_dir "$OUT_DIR" --emulator_shuffle "$SHUFFLE")
[ "$DEVKIT" != "none" ] && RUN_ARGS+=(--speck_devkit "$DEVKIT")
[ -n "${N_TRAIN:-}" ] && RUN_ARGS+=(--n_train "$N_TRAIN")
[ -n "${N_VAL:-}" ] && RUN_ARGS+=(--n_val "$N_VAL")
[ -n "${N_TEST:-}" ] && RUN_ARGS+=(--n_test "$N_TEST")
[ -n "${EPOCHS:-}" ] && RUN_ARGS+=(--epochs "$EPOCHS")
[ -n "${INIT:-}" ] && RUN_ARGS+=(--init "$INIT")
for regime in $REGIMES; do        # one process per regime: a failure leaves the others' results
    echo -e "\n--- $(date '+%T') ${regime}"
    python3 lab.py run "${RUN_ARGS[@]}" --regimes "$regime" || echo "WARNING: ${regime} failed (see above); continuing"
done

# --- 3. table and figures over every regime ---
echo -e "\n=== 3/3 summary"
python3 lab.py summary --out_dir "$OUT_DIR"
echo -e "\nDone $(date '+%F %T'): ${OUT_DIR}/results.json, lab_rmse.png, lab_rates.png"
