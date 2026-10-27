#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J SPECK_LAB
#SBATCH -o lab_logs/SPECK_LAB_%A_%a.out
#SBATCH -e lab_logs/SPECK_LAB_%A_%a.out
#
# Train the speck_lab regimes on the cluster, one array task per regime (the
# chip is added later on the Speck laptop: lab.py run --speck_devkit, which
# reuses these models). Submit from main/speck_lab:
#
#   mkdir -p lab_logs
#   sbatch --export=ALL,SNN_DATASET=$DATA_ROOT/snn_datasets/hkm/jenkins/mua/<session>,OUT_DIR=lab_runs/jenkins \
#       --array=0-5 run_lab.sbatch
#
# REGIMES (space-separated, default below) picks the regimes; task i trains
# the i-th. LAB_ARGS passes more lab.py run options, e.g. "--n_train 120 --epochs 15".
# When every task has finished: python lab.py summary --out_dir <OUT_DIR>

set -eo pipefail
: "${SNN_DATASET:?set SNN_DATASET to a session directory with train/ and test/}"
: "${OUT_DIR:?set OUT_DIR}"
read -r -a REGIME_LIST <<< "${REGIMES:-baseline rate10 rate30 rate10_sparse rate10_single rate10_emusel}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK" -ge "${#REGIME_LIST[@]}" ]; then
    echo "Array task $TASK has no regime (${#REGIME_LIST[@]} given); exiting."
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snn_speck
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Regime ${REGIME_LIST[$TASK]} on $SNN_DATASET -> $OUT_DIR"
# shellcheck disable=SC2086
python3 lab.py run --snn_dataset "$SNN_DATASET" --out_dir "$OUT_DIR" --regimes "${REGIME_LIST[$TASK]}" ${LAB_ARGS:-}
