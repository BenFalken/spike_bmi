#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH -J TEST_ALL_DECODERS
#SBATCH -o logs/TEST_ALL_DECODERS_%A_%a.out
#SBATCH -e logs/TEST_ALL_DECODERS_%A_%a.out
#
# Re-runs each decoder from its cached checkpoint (WF/KF/MLP/LSTM/QRNN/SNN)
# to gather RMSE, latency, energy, etc. all directly comparable against
# each other. One SESSION per array task -- for real SLURM-level
# parallelism across nodes, not test_all_decoders.py's own --multi_session
# mode, which runs sequentially within a single job.
#
# Runs independently per (EXPERIMENT, SUBJECT) -- pass both via --export,
# same convention as every other array script in this project. Results
# land under results/test_all_decoders/{experiment}/{subject}/, so
# different subjects/experiments can never collide or overwrite each
# other's output.
#
# Run combine_session_metrics.py after this completes to produce the
# full cross-session report.
#
# Submit with:
#   mkdir -p logs
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/bmi/indy/per_session -maxdepth 1 -mindepth 1 -type d | wc -l)
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=jenkins --array=0-6 run_test_all_decoders_array.sbatch                     # validate one session first
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=nitschke --array=0-6 run_test_all_decoders_array.sbatch            # the rest

set -eo pipefail

# --- Activate Environment ---
deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# --- Resolve Experiment/Subject ---
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_test_all_decoders_array.sbatch" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_test_all_decoders_array.sbatch" >&2
    exit 1
fi

# --- Customizable Src/Dest Paths -- matching this project's real,
# current directory conventions. RESULTS root moved to scratch (was
# under /users/bfalkenb/spike_bmi_main, the project directory) --
# confirmed directly against a real run that results/model_cache were
# actually landing on scratch, not the project folder, so everything
# under "results" now derives from BMI_DATA_ROOT to match. ---
BMI_DATA_ROOT="/users/bfalkenb/scratch/bfalkenb/data"
CHECKPOINTS_DIR="${BMI_DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/per_session"
DATASET_ROOT="${BMI_DATA_ROOT}/dataset/${EXPERIMENT}/${SUBJECT}/mua"
MODEL_CACHE_ROOT="${BMI_DATA_ROOT}/results/model_cache/${EXPERIMENT}/${SUBJECT}"
# SNN dataset source is genuinely different per experiment, not a shared
# convention: bmi still uses mua_8_group (256ms-windowed, group_size=8
# chunked, per this project's own settled convention -- unchanged).
# hkm switched entirely to whole-trial processing (see
# make_snn_dataset_whole_trial.py's own module docstring for why: a
# large fraction of real hkm trials are shorter than even one 256ms
# window, so windowing/grouping them was silently discarding real data)
# -- its own data lives directly under .../snn_datasets/hkm/{subject}/mua/
# {session}/, no group_size suffix at all. A hardcoded mua_8_group here
# regardless of EXPERIMENT would point every hkm session at a directory
# that simply doesn't exist, failing outright (confirmed this WAS the
# case in an earlier revision of this script, before this fix).
if [ "$EXPERIMENT" = "hkm" ]; then
    SNN_DATASET_ROOT="${BMI_DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"
else
    SNN_DATASET_ROOT="${BMI_DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
fi
RESULTS_DIR="${BMI_DATA_ROOT}/results/test_all_decoders/${EXPERIMENT}/${SUBJECT}"
FEATURE="mua"
DECODERS="mlp,lstm,qrnn,kf,wf,snn"
TEST_FRAC=0.1
TRAIN_DURATIONS="1,2,3,4,5,6,7,8,9,10"
LAST_DURATION_MIN=10

# --continuous-snn-test-stream is HKM-only and opt-in -- unset/empty
# (the default) leaves it off entirely, matching every existing
# submission's current behavior exactly (reset SNN state at every test
# trial). Set to any non-empty value via --export to turn it on for a
# given submission, e.g.:
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=jenkins,CONTINUOUS_SNN_TEST_STREAM=1 \
#       --array=0-6 run_test_all_decoders_array.sbatch
# Silently inert for EXPERIMENT=bmi even if set -- see
# test_all_decoders.py's own predict_snn_trial() docstring for why that's
# guaranteed, not just expected.
CONTINUOUS_SNN_TEST_STREAM="${CONTINUOUS_SNN_TEST_STREAM:-}"
EXTRA_SNN_ARGS=()
if [ -n "$CONTINUOUS_SNN_TEST_STREAM" ]; then
    EXTRA_SNN_ARGS+=(--continuous-snn-test-stream)
fi

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "ERROR: $CHECKPOINTS_DIR does not exist -- check EXPERIMENT/SUBJECT." >&2
    exit 1
fi

# --- Resolve Session -- only sessions with a real best_model_weights.pth
# are counted, matching this project's real, current per_session layout
# (no nested config-name subdirectory anymore). ---
mapfile -t ALL_CANDIDATE_DIRS < <(find "$CHECKPOINTS_DIR" -maxdepth 1 -mindepth 1 -type d | sort)
SESSION_IDS=()
for d in "${ALL_CANDIDATE_DIRS[@]}"; do
    if [ -f "${d}/best_model_weights.pth" ]; then
        SESSION_IDS+=("$(basename "$d")")
    fi
done
N_SESSIONS=${#SESSION_IDS[@]}
if [ "$N_SESSIONS" -eq 0 ]; then
    echo "ERROR: found zero sessions with best_model_weights.pth under $CHECKPOINTS_DIR" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no corresponding session "
    echo "(only $N_SESSIONS found) -- exiting cleanly."
    exit 0
fi

SESSION_ID="${SESSION_IDS[$SLURM_ARRAY_TASK_ID]}"

# --- Per-session Src/Dest Paths ---
INPUT_FILEPATH="${DATASET_ROOT}/${SESSION_ID}_binning.h5"
# NOTE: no trailing /${FEATURE} here -- test_all_decoders.py appends
# args.feature onto model_dir ITSELF internally (bundle_dir =
# os.path.join(model_dir, feature), confirmed directly in its own
# load_dl_decoder()/load_kf_decoder()/load_wf_decoder()); passing an
# already-feature-suffixed path here produced a real, confirmed
# ".../mua/mua/..." double segment and every cache lookup missed.
MODEL_DIR="${MODEL_CACHE_ROOT}/${SESSION_ID}"
SNN_CHECKPOINT_PATH="${CHECKPOINTS_DIR}/${SESSION_ID}/best_model_weights.pth"
SNN_DATASET_PATH="${SNN_DATASET_ROOT}/${SESSION_ID}"
METRICS_SAVE_PATH="${RESULTS_DIR}/${SESSION_ID}_metrics.json"
LAST_DURATION_FILE="${RESULTS_DIR}/${SESSION_ID}_metrics_${LAST_DURATION_MIN}min.json"

if [ ! -f "$INPUT_FILEPATH" ]; then
    echo "ERROR: $INPUT_FILEPATH does not exist -- WF/KF/ANN eval for this session may "
    echo "       not have completed yet." >&2
    exit 1
fi
if [ ! -d "$SNN_DATASET_PATH" ]; then
    echo "ERROR: $SNN_DATASET_PATH does not exist -- check that the SNN dataset has "
    echo "       actually been built for this session (mua_8_group for bmi, whole-trial "
    echo "       mua for hkm -- see SNN_DATASET_ROOT's own construction above)." >&2
    exit 1
fi

# --- Start Testing Decoders ---
echo "Array task $SLURM_ARRAY_TASK_ID -> experiment=$EXPERIMENT subject=$SUBJECT session=$SESSION_ID " \
     "($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS})"
echo "  input=$INPUT_FILEPATH snn_checkpoint=$SNN_CHECKPOINT_PATH"
mkdir -p "$RESULTS_DIR"

# --- Full-model evaluation (NO --train_durations) ---
# test_all_decoders.py treats --train_durations as an either/or choice,
# not additive (duration_minutes_list = the parsed list WHEN
# --train_durations is set, else just [None] for the untagged full-data
# case -- confirmed directly in its own args-handling) -- so the full,
# non-duration-capped evaluation needs its OWN, separate invocation here,
# same two-call pattern single_subject_pipeline.py already uses for
# WF/KF (full-data eval, then duration sweep, as two separate run_cmd()
# calls). Checked and skipped INDEPENDENTLY of the duration sweep below,
# so a completed sweep with a still-missing full-data file (or vice
# versa) is detected and fixed correctly, not silently treated as "this
# session is done."
if [ -f "$METRICS_SAVE_PATH" ]; then
    echo "[skip] ${SESSION_ID}: full-model $METRICS_SAVE_PATH already exists -- delete it "
    echo "       first if you actually want to re-run the full-data evaluation."
else
    python3 ../inference/test_all_decoders.py \
        --experiment "$EXPERIMENT" \
        --input_filepath "$INPUT_FILEPATH" \
        --model_dir "$MODEL_DIR" \
        --snn_checkpoint_path "$SNN_CHECKPOINT_PATH" \
        --snn_dataset_path "$SNN_DATASET_PATH" \
        --metrics_save_path "$METRICS_SAVE_PATH" \
        --feature "$FEATURE" \
        --decoders "$DECODERS" \
        --test_frac "$TEST_FRAC" \
        "${EXTRA_SNN_ARGS[@]}"
fi

# --- Duration sweep ---
if [ -f "$LAST_DURATION_FILE" ]; then
    echo "[skip] ${SESSION_ID}: $LAST_DURATION_FILE already exists (the whole "
    echo "       --train_durations sweep for this session completed) -- delete the "
    echo "       ${SESSION_ID}_metrics_*.json files first if you actually want to "
    echo "       re-run this session."
else
    python3 ../inference/test_all_decoders.py \
        --experiment "$EXPERIMENT" \
        --input_filepath "$INPUT_FILEPATH" \
        --model_dir "$MODEL_DIR" \
        --snn_checkpoint_path "$SNN_CHECKPOINT_PATH" \
        --snn_dataset_path "$SNN_DATASET_PATH" \
        --metrics_save_path "$METRICS_SAVE_PATH" \
        --feature "$FEATURE" \
        --decoders "$DECODERS" \
        --test_frac "$TEST_FRAC" \
        --train_durations "$TRAIN_DURATIONS" \
        "${EXTRA_SNN_ARGS[@]}"
fi
