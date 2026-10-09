#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -J HKM_DIAGNOSE
#SBATCH -o hkm_diagnose_logs/HKM_DIAGNOSE_%j.out
#SBATCH -e hkm_diagnose_logs/HKM_DIAGNOSE_%j.out
#
# Runs diagnose_hkm_sessions.py over every Nitschke session (default).
#
# Submit with:
#   mkdir -p hkm_diagnostics hkm_diagnose_logs
#   sbatch run_diagnose_hkm_sessions.sbatch
#
# Other subject / extra options:
#   sbatch --export=ALL,SUBJECT=jenkins run_diagnose_hkm_sessions.sbatch
#   sbatch --export=ALL,EXTRA_ARGS="--v-lo -300 --v-hi 300 --dt-ms 10" run_diagnose_hkm_sessions.sbatch
#
# Outputs land in hkm_diagnostics/<SUBJECT>/ (figures, diagnostics.json, diagnostics.txt).

set -eo pipefail

# --- Activate Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck

# --- CPU-only thread capping ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Paths ---
SUBJECT="${SUBJECT:-nitschke}"
DATA_ROOT="/users/bfalkenb/scratch/bfalkenb/data/snn_datasets/hkm/${SUBJECT}/mua"
OUT_DIR="hkm_diagnostics/${SUBJECT}"
SCRIPT="${SCRIPT:-../preprocessing_training/diagnose_hkm_sessions.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: $DATA_ROOT does not exist -- check SUBJECT." >&2
    exit 1
fi
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found (override with SCRIPT=...)." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Diagnosing subject=$SUBJECT data-root=$DATA_ROOT out-dir=$OUT_DIR"

# shellcheck disable=SC2086
python3 "$SCRIPT" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS
