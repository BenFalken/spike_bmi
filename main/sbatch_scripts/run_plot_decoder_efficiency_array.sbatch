#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J DECODER_EFFICIENCY_PROFILE
#SBATCH -o logs/DECODER_EFFICIENCY_PROFILE_%A_%a.out
#SBATCH -e logs/DECODER_EFFICIENCY_PROFILE_%A_%a.out
#
# ONE array task PER SESSION -- profiles per-sample latency, per-sample
# energy, and param_count for every decoder in that session via
# plot_decoder_efficiency.py --profile_save_path --skip_figure (no
# per-session figure needed; only the aggregate figure across every
# session's profile matters here -- see
# plot_decoder_efficiency_aggregate.py).
#
# Runs independently per (EXPERIMENT, SUBJECT) -- pass both via --export,
# same convention as every other array script in this project.
#
# --cpus-per-task MATCHES run_test_all_decoders_array.sbatch's own value
# DELIBERATELY, not by coincidence -- EnergyMeter's proxy_psutil
# fallback, and the OMP_NUM_THREADS/etc. thread-pinning both scripts now
# share via test_all_decoders.py's own top-of-file setup (imported here
# for EnergyMeter + load_snn_model), key off SLURM_CPUS_PER_TASK.
# Profiling under a DIFFERENT allocation than your evaluation run would
# NOT be the equal-footing comparison this whole rework was for -- if
# you change one script's --cpus-per-task, change both.
#
# REQUIRES combined_metrics.json (RMSE) to already exist for this
# EXPERIMENT/SUBJECT -- run_test_all_decoders_array.sbatch +
# combine_session_metrics.py must have completed first; this script
# filters records to decoders that have an RMSE entry there.
#
# Session-specific paths match run_test_all_decoders_array.sbatch's own
# real, current conventions:
#   dataset file    : {DATASET_ROOT}/{session_id}_binning.h5
#   model cache     : {MODEL_CACHE_ROOT}/{session_id}   (NO trailing
#                      /{feature} -- plot_decoder_efficiency.py appends
#                      that itself internally, same as test_all_decoders.py's
#                      own load_kf_decoder()/load_wf_decoder(); passing
#                      an already-suffixed path here would reproduce the
#                      same ".../mua/mua/..." double-segment bug found
#                      and fixed in run_test_all_decoders_array.sbatch)
#   snn checkpoint  : {CHECKPOINTS_DIR}/{session_id}/best_model_weights.pth
#   snn dataset     : {SNN_DATASET_ROOT}/{session_id}
#   profile output  : {PROFILES_DIR}/{session_id}_profile.json
#
# Submit with:
#   mkdir -p logs
#   N=$(find /users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/bmi/indy/per_session -maxdepth 1 -mindepth 1 -type d | wc -l)
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0 run_plot_decoder_efficiency_array.sbatch                     # validate one session first
#   sbatch --export=EXPERIMENT=bmi,SUBJECT=loco --array=0-9 run_plot_decoder_efficiency_array.sbatch            # the rest
#   sbatch --export=EXPERIMENT=hkm,SUBJECT=jenkins --array=0-3 run_plot_decoder_efficiency_array.sbatch         # hkm (4 sessions)
#
# Then, once every array task has finished, build the aggregate figures
# (plot_decoder_efficiency_aggregate.py already takes explicit paths via
# argparse, no changes needed there):
#   python plot_decoder_efficiency_aggregate.py \
#       --profiles_dir results/decoder_efficiency_profiles/bmi/indy \
#       --combined_metrics_path results/test_all_decoders/bmi/indy/combined_metrics.json \
#       --decoders kf,wf,lstm,qrnn,snn \
#       --save_path results/decoder_efficiency_aggregate_bmi_indy.png \
#       --energy_save_path results/decoder_energy_aggregate_bmi_indy.png

set -eo pipefail  # -u intentionally omitted -- see the earlier note on venv activation

# --- Activate Environment ---
deactivate 2>/dev/null || true
source "$HOME/ann-env/bin/activate"

# --- Resolve Experiment/Subject ---
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT is not set (expected 'bmi' or 'hkm') -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_plot_decoder_efficiency_array.sbatch" >&2
    exit 1
fi
if [ -z "$SUBJECT" ]; then
    echo "ERROR: SUBJECT is not set -- submit with, e.g., "
    echo "       sbatch --export=EXPERIMENT=bmi,SUBJECT=indy --array=0-N run_plot_decoder_efficiency_array.sbatch" >&2
    exit 1
fi

# --- Fixed settings -- matching run_test_all_decoders_array.sbatch's own real,
# current conventions (BMI_DATA_ROOT-rooted, not the project directory) ---
BMI_DATA_ROOT="/users/bfalkenb/scratch/bfalkenb/data"
CHECKPOINTS_DIR="${BMI_DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}/per_session"
DATASET_ROOT="${BMI_DATA_ROOT}/dataset/${EXPERIMENT}/${SUBJECT}/mua"
MODEL_CACHE_ROOT="${BMI_DATA_ROOT}/results/model_cache/${EXPERIMENT}/${SUBJECT}"
# bmi: 256ms-windowed, group_size=8 chunked (mua_8_group). hkm: whole trials, no group suffix --
# same split run_test_all_decoders_array.sbatch makes.
if [ "$EXPERIMENT" = "hkm" ]; then
    SNN_DATASET_ROOT="${BMI_DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua"
else
    SNN_DATASET_ROOT="${BMI_DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_8_group"
fi
COMBINED_METRICS_PATH="${BMI_DATA_ROOT}/results/test_all_decoders/${EXPERIMENT}/${SUBJECT}/combined_metrics.json"
PROFILES_DIR="${BMI_DATA_ROOT}/results/decoder_efficiency_profiles/${EXPERIMENT}/${SUBJECT}"
FEATURE="mua"
DECODERS="kf,wf,lstm,qrnn,snn"
TEST_FRAC=0.1
N_ENERGY_REPEATS=10

if [ ! -f "$COMBINED_METRICS_PATH" ]; then
    echo "ERROR: $COMBINED_METRICS_PATH not found -- run "
    echo "       run_test_all_decoders_array.sbatch + combine_session_metrics.py first "
    echo "       (RMSE has to exist before this script can filter records by it)." >&2
    exit 1
fi

# --- Resolve Session -- only sessions with a real best_model_weights.pth
# are counted, matching run_test_all_decoders_array.sbatch's own real,
# flat per_session layout (no nested config-name subdirectory). ---
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

DATASET_FILEPATH="${DATASET_ROOT}/${SESSION_ID}_binning.h5"
MODEL_DIR="${MODEL_CACHE_ROOT}/${SESSION_ID}"
SNN_CHECKPOINT_PATH="${CHECKPOINTS_DIR}/${SESSION_ID}/best_model_weights.pth"
SNN_DATASET_PATH="${SNN_DATASET_ROOT}/${SESSION_ID}"
PROFILE_SAVE_PATH="${PROFILES_DIR}/${SESSION_ID}_profile.json"

if [ ! -f "$DATASET_FILEPATH" ]; then
    echo "ERROR: $DATASET_FILEPATH does not exist -- WF/KF/ANN eval for this session may "
    echo "       not have completed yet." >&2
    exit 1
fi
if [ ! -d "$SNN_DATASET_PATH" ]; then
    echo "ERROR: $SNN_DATASET_PATH does not exist -- check that the SNN datasets (mua_8_group "
    echo "       for bmi, whole-trial mua for hkm) have been built for this session." >&2
    exit 1
fi

if [ -f "$PROFILE_SAVE_PATH" ]; then
    echo "[skip] ${SESSION_ID}: $PROFILE_SAVE_PATH already exists -- delete it first "
    echo "       if you actually want to re-profile this session."
    exit 0
fi

echo "Array task $SLURM_ARRAY_TASK_ID -> experiment=$EXPERIMENT subject=$SUBJECT session=$SESSION_ID " \
     "($((SLURM_ARRAY_TASK_ID+1))/${N_SESSIONS})"
echo "  dataset=$DATASET_FILEPATH snn_checkpoint=$SNN_CHECKPOINT_PATH"
mkdir -p "$PROFILES_DIR"

python3 ../inference/plot_decoder_efficiency.py \
    --experiment "$EXPERIMENT" \
    --dataset_filepath "$DATASET_FILEPATH" \
    --model_dir "$MODEL_DIR" \
    --feature "$FEATURE" \
    --test_frac "$TEST_FRAC" \
    --combined_metrics_path "$COMBINED_METRICS_PATH" \
    --decoders "$DECODERS" \
    --snn_checkpoint_path "$SNN_CHECKPOINT_PATH" \
    --snn_dataset_path "$SNN_DATASET_PATH" \
    --n_energy_repeats "$N_ENERGY_REPEATS" \
    --profile_save_path "$PROFILE_SAVE_PATH" \
    --skip_figure
