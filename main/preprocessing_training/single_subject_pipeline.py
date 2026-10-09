#!/usr/bin/env python
"""
Single-session BMI decoding pipeline.

For one recording session this script builds the decoding datasets, then
evaluates the classical (KF, WF) and deep-learning (LSTM, QRNN) decoders,
both on all available training data and as a function of training
duration, and saves quick-look comparison figures. Each stage is a separate
script in this directory, run as a subprocess:

  1. process_data.py        raw .mat -> spike trains + kinematics (.h5)
  2. make_dataset.py        ANN dataset: dense 256 ms windows at 4 ms steps
     make_snn_dataset.py    SNN dataset: back-to-back 256 ms spike rasters
     check_datasets.py      sanity checks on both
     export_snn_pkl.py      SNN windows -> per-window train/test .pkl files
     combine_snn_dataset.py SNN windows -> long (group_size-window) trials
  3. eval_wf_decoder.py, eval_kf_decoder.py
  4. eval_dl_decoders.py    (hyperparameter JSONs written from DL_HYPERPARAMS)
  5. comparison figures

SNN training is separate (snn_training/train_snn.py, driven by
sbatch_scripts/run_snn_*.sbatch), and the cross-decoder comparison
including the SNN is inference/test_all_decoders.py.

Two kinds of session are supported:
  - Continuous sessions (experiment "bmi", raw_stem like 'indy_20160407_02'):
    process_session() runs every stage, starting from the raw .mat download.
  - Trial-structured NWB sessions (experiment "hkm", raw_stem like
    'sub-Nitschke_ses-20090812_behavior+ecephys'): the datasets come from
    nwb_conversion/run_nwb_pipeline.sh, so process_nwb_session() runs only
    stages 3-5.

Every stage skips outputs that already exist (unless --overwrite), so an
interrupted run can simply be restarted.

Storage locations are set by the environment variable BMI_DATA_ROOT; see
the path helpers below for the layout.

Usage:
    python single_subject_pipeline.py --raw_stem indy_20160407_02
    python single_subject_pipeline.py --raw_stem SESSION --nwb_dataset_filepath SESSION_binning.h5
"""

import json
import os
import re
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.utils import customize_plot

# Keep log output in order when stdout is redirected to a file (e.g. under SLURM).
sys.stdout.reconfigure(line_buffering=True)

BMI_DATA_ROOT = os.environ.get("BMI_DATA_ROOT", "/users/bfalkenb/scratch/bfalkenb/data")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


# ============================================================================
# Configuration
# ============================================================================

RAW_STEM = "indy_20160407_02"   # default session
FEATURE = "mua"                  # "mua" (threshold crossings per channel) or "sua" (sorted units)
METHOD = "binning"               # spike-rate estimation method for the ANN dataset
TEST_FRAC = 0.1                  # chronologically last fraction held out as test
DURATIONS = tuple(range(1, 11))  # training durations (minutes) for the duration sweep
MIN_TRAIN_SIZE = 0.45            # minimum first-CV-fold training size (fraction of rows)

# Windowing. Both datasets use 256 ms (65-sample) windows. The ANN windows
# advance by one 4 ms sample, the SNN windows sit back to back.
WDW_TIME = 0.256
ANN_OL_TIME = 0.252   # ANN step = WDW_TIME - ANN_OL_TIME = 4 ms
ANN_STEP_MS = 4.0
SNN_OL_TIME = 0.0

# Wiener filter. 15 taps as in the reference paper. HKM sessions have 192
# channels, and 15 taps there exhausts memory, so they use 8.
WF_TAP = 15
WF_TAP_HKM = 8
WF_REG_TYPE = "l2"
WF_REG_ALPHA = 1.0    # not cross-validated

# LSTM/QRNN hyperparameters, taken from the reference paper's reported
# values and used as-is for every session and feature.
DL_DECODERS = ("lstm", "qrnn")
DL_HYPERPARAMS = {
    "lstm": {"units": 200, "epochs": 5, "batch_size": 32, "dropout": 0.5, "learning_rate": 0.004},
    "qrnn": {"units": 400, "epochs": 2, "batch_size": 64, "dropout": 0.5, "learning_rate": 0.0015},
}
DL_FIXED = {"timesteps": 5, "n_layers": 1, "optimizer": "Adam"}

ALL_DECODERS = ("kf", "wf") + DL_DECODERS
DECODER_COLORS = {
    "kf": "purple", "wf": "goldenrod", "mlp": "crimson",
    "lstm": "darkorange", "qrnn": "seagreen", "snn": "royalblue",
}


# ============================================================================
# Paths
# ============================================================================
# Everything lives under BMI_DATA_ROOT, namespaced by experiment and subject:
#   raw/{exp}/{subj}/{raw_stem}.mat
#   processed/{exp}/{subj}/{feature}/{raw_stem}.h5
#   dataset/{exp}/{subj}/{feature}/{raw_stem}_{method}.h5 and {raw_stem}_snn.h5
#   snn_datasets/{exp}/{subj}/{feature}/{raw_stem}/{train,test}/*.pkl
#   snn_datasets/{exp}/{subj}/{feature}_{group_size}_group/{raw_stem}/{train,test}/*.pkl
#   results/{exp}/{subj}/decoder/{raw_stem}_{feature}_{method}_{decoder}[_{N}min].h5
#   results/model_cache/{exp}/{subj}/{raw_stem}/{feature}/   (final model bundles)
#   params/{exp}/{subj}/   (DL hyperparameter JSONs)
#   figures/{exp}/{subj}/

_NWB_RAW_STEM_RE = re.compile(r"^sub-([A-Za-z]+)_ses-")


def experiment_and_subject_from_raw_stem(raw_stem):
    """'indy_20160407_02' -> ('bmi', 'indy');
    'sub-Nitschke_ses-20090812_behavior+ecephys' -> ('hkm', 'nitschke')."""
    m = _NWB_RAW_STEM_RE.match(raw_stem)
    if m:
        return "hkm", m.group(1).lower()
    subject = raw_stem.split("_")[0]
    if not subject:
        raise ValueError(f"Cannot parse experiment/subject from raw_stem={raw_stem!r}; expected "
                         f"'sub-{{Subject}}_ses-...' or '{{subject}}_{{date}}_{{index}}'.")
    return "bmi", subject


def _makedirs(*parts):
    path = os.path.join(BMI_DATA_ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def raw_dir(raw_stem):
    return _makedirs("raw", *experiment_and_subject_from_raw_stem(raw_stem))


def feature_dirs(raw_stem, feature):
    exp_subj = experiment_and_subject_from_raw_stem(raw_stem)
    return {
        "processed": _makedirs("processed", *exp_subj, feature),
        "dataset": _makedirs("dataset", *exp_subj, feature),
        "snn_pkl_root": _makedirs("snn_datasets", *exp_subj, feature),
    }


def subject_dirs(raw_stem):
    exp_subj = experiment_and_subject_from_raw_stem(raw_stem)
    return {
        "results_decoder": _makedirs("results", *exp_subj, "decoder"),
        "params": _makedirs("params", *exp_subj),
        "figures": _makedirs("figures", *exp_subj),
    }


def model_cache_dir(raw_stem, feature=FEATURE):
    return _makedirs("results", "model_cache", *experiment_and_subject_from_raw_stem(raw_stem),
                     raw_stem, feature)


def result_path(raw_stem, decoder, feature, method, minutes=None):
    suffix = "" if minutes is None else f"_{minutes:g}min"
    return os.path.join(subject_dirs(raw_stem)["results_decoder"],
                        f"{raw_stem}_{feature}_{method}_{decoder}{suffix}.h5")


def dl_config_path(raw_stem, decoder, feature, method):
    return os.path.join(subject_dirs(raw_stem)["params"], f"{raw_stem}_{feature}_{method}_{decoder}.json")


# ============================================================================
# Helpers
# ============================================================================

def run_cmd(args, description):
    """Run a stage script unbuffered (-u, so its progress shows up in logs
    immediately); raises CalledProcessError if it fails."""
    print(f"\n=== {description} ===")
    print(" ".join(args))
    subprocess.run([args[0], "-u"] + args[1:], check=True)


def _skip_if_exists(path, overwrite, description):
    if not overwrite and os.path.exists(path):
        print(f"[skip] {description}: {path} already exists")
        return True
    return False


# ============================================================================
# Stage 1-2: data acquisition and dataset construction
# ============================================================================

def download_raw_data(raw_stem, overwrite=False):
    """Fetch {raw_stem}.mat from the Zenodo record of the O'Doherty et al.
    reaching dataset."""
    url = f"https://zenodo.org/record/3854034/files/{raw_stem}.mat"
    raw_filepath = os.path.join(raw_dir(raw_stem), f"{raw_stem}.mat")
    if _skip_if_exists(raw_filepath, overwrite, "raw download"):
        return raw_filepath
    print(f"\n=== Downloading {url} ===")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(raw_filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return raw_filepath


def process_raw_data(raw_stem, feature=FEATURE, overwrite=False):
    raw_filepath = os.path.join(raw_dir(raw_stem), f"{raw_stem}.mat")
    processed_filepath = os.path.join(feature_dirs(raw_stem, feature)["processed"], f"{raw_stem}.h5")
    if not _skip_if_exists(processed_filepath, overwrite, "process_data"):
        run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "process_data.py"),
                 "--input_filepath", raw_filepath,
                 "--output_filepath", processed_filepath],
                "process_data.py")
    return processed_filepath


def build_ann_dataset(raw_stem, processed_filepath, feature=FEATURE, method=METHOD, overwrite=False):
    dataset_filepath = os.path.join(feature_dirs(raw_stem, feature)["dataset"], f"{raw_stem}_{method}.h5")
    if not _skip_if_exists(dataset_filepath, overwrite, "make_dataset"):
        run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "make_dataset.py"),
                 "--input_filepath", processed_filepath,
                 "--output_filepath", dataset_filepath,
                 "--method", method,
                 "--wdw_time", str(WDW_TIME),
                 "--ol_time", str(ANN_OL_TIME)],
                "make_dataset.py")
    return dataset_filepath


def build_snn_dataset(raw_stem, processed_filepath, feature=FEATURE, overwrite=False):
    """Build the SNN .h5, validate it alongside the ANN dataset, and export
    per-window .pkl files. Returns (snn_dataset_filepath, snn_pkl_path)."""
    dirs = feature_dirs(raw_stem, feature)
    snn_dataset_filepath = os.path.join(dirs["dataset"], f"{raw_stem}_snn.h5")
    if not _skip_if_exists(snn_dataset_filepath, overwrite, "make_snn_dataset"):
        run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "make_snn_dataset.py"),
                 "--input_filepath", processed_filepath,
                 "--output_filepath", snn_dataset_filepath,
                 "--feature", feature,
                 "--wdw_time", str(WDW_TIME),
                 "--ol_time", str(SNN_OL_TIME)],
                "make_snn_dataset.py")

    run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "check_datasets.py"),
             "--dataset_dirname", dirs["dataset"],
             "--raw_stem", raw_stem,
             "--methods", METHOD],
            "check_datasets.py")

    snn_pkl_path = os.path.join(dirs["snn_pkl_root"], raw_stem)
    if not _skip_if_exists(os.path.join(snn_pkl_path, "test"), overwrite, "export_snn_pkl"):
        run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "export_snn_pkl.py"),
                 "--input_filepath", snn_dataset_filepath,
                 "--output_path", snn_pkl_path,
                 "--test_frac", str(TEST_FRAC)],
                "export_snn_pkl.py")
    return snn_dataset_filepath, snn_pkl_path


def build_combined_snn_dataset(raw_stem, snn_dataset_filepath, feature=FEATURE,
                               group_size=8, test_frac=TEST_FRAC, overwrite=False,
                               discard_remainder=True):
    """Group group_size consecutive SNN windows into longer training trials
    (see combine_snn_dataset.py). discard_remainder drops the shorter final
    train trial so trials can be batched."""
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    output_dir = os.path.join(BMI_DATA_ROOT, "snn_datasets", experiment, subject,
                              f"{feature}_{group_size}_group")
    marker_path = os.path.join(output_dir, raw_stem, "train", "0.pkl")
    if _skip_if_exists(marker_path, overwrite, f"combine_snn_dataset (group_size={group_size})"):
        return output_dir

    cmd = [PYTHON, os.path.join(SCRIPT_DIR, "combine_snn_dataset.py"),
           "--h5-path", snn_dataset_filepath,
           "--session-id", raw_stem,
           "--group-size", str(group_size),
           "--test-frac", str(test_frac),
           "--output-dir", output_dir]
    if discard_remainder:
        cmd.append("--discard-train-remainder")
    run_cmd(cmd, f"combine_snn_dataset.py (group_size={group_size})")
    return output_dir


# ============================================================================
# Stage 3-4: decoder evaluation
# ============================================================================

def _run_decoder_eval(script, decoder, raw_stem, dataset_filepath, feature, method, extra_args,
                      overwrite, n_train_override, gap_samples):
    """Run an eval script twice: once on all training data, once sweeping
    DURATIONS. The script skips any result or model bundle that already
    exists, so this is cheap to repeat."""
    args = [PYTHON, os.path.join(SCRIPT_DIR, script),
            "--input_filepath", dataset_filepath,
            "--output_filepath", result_path(raw_stem, decoder, feature, method),
            "--model_dir", model_cache_dir(raw_stem, feature),
            "--test_frac", str(TEST_FRAC),
            "--feature", feature,
            "--wdw_time", str(WDW_TIME),
            "--step_ms", str(ANN_STEP_MS),
            "--min_train_size", str(MIN_TRAIN_SIZE),
            *extra_args]
    if n_train_override is not None:
        args += ["--n_train_override", str(n_train_override)]
    if gap_samples is not None:
        args += ["--gap_samples", str(gap_samples)]
    if overwrite:
        args.append("--overwrite")

    run_cmd(args, f"{decoder.upper()} full-data eval")
    run_cmd(args + ["--train_durations", ",".join(str(d) for d in DURATIONS)],
            f"{decoder.upper()} duration sweep")
    return result_path(raw_stem, decoder, feature, method)


def eval_wf(raw_stem, dataset_filepath, wf_tap=WF_TAP, feature=FEATURE, method=METHOD,
            overwrite=False, n_train_override=None, gap_samples=None):
    return _run_decoder_eval(
        "eval_wf_decoder.py", "wf", raw_stem, dataset_filepath, feature, method,
        ["--timesteps", str(wf_tap), "--reg_type", WF_REG_TYPE, "--reg_alpha", str(WF_REG_ALPHA)],
        overwrite, n_train_override, gap_samples)


def eval_kf(raw_stem, dataset_filepath, feature=FEATURE, method=METHOD,
            overwrite=False, n_train_override=None, gap_samples=None):
    return _run_decoder_eval("eval_kf_decoder.py", "kf", raw_stem, dataset_filepath, feature,
                             method, [], overwrite, n_train_override, gap_samples)


def write_dl_decoder_configs(raw_stem, decoders=DL_DECODERS, feature=FEATURE, method=METHOD,
                             overwrite=False):
    """Write each DL decoder's hyperparameter JSON (read by eval_dl_decoders.py)."""
    for decoder in decoders:
        config_filepath = dl_config_path(raw_stem, decoder, feature, method)
        if _skip_if_exists(config_filepath, overwrite, f"{decoder.upper()} config"):
            continue
        config = {**DL_FIXED, **DL_HYPERPARAMS[decoder]}
        with open(config_filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote {decoder.upper()} config to {config_filepath}: {config}")


def eval_dl_decoders(raw_stem, dataset_filepath, decoders=DL_DECODERS, feature=FEATURE,
                     method=METHOD, overwrite=False, n_train_override=None, gap_samples=None):
    for decoder in decoders:
        _run_decoder_eval(
            "eval_dl_decoders.py", decoder, raw_stem, dataset_filepath, feature, method,
            ["--decoder", decoder,
             "--config_filepath", dl_config_path(raw_stem, decoder, feature, method)],
            overwrite, n_train_override, gap_samples)


# ============================================================================
# Stage 5: quick-look figures
# ============================================================================

def load_full_results(raw_stem, decoders=ALL_DECODERS, feature=FEATURE, method=METHOD):
    """{decoder: (rmse_test_folds, cc_test_folds)} for the full-data runs."""
    results = {}
    for decoder in decoders:
        path = result_path(raw_stem, decoder, feature, method)
        if not os.path.exists(path):
            print(f"[skip] {decoder.upper()}: no full-data result at {path}")
            continue
        with h5py.File(path, "r") as f:
            results[decoder] = (f["rmse_test_folds"][()], f["cc_test_folds"][()])
    return results


def load_duration_results(raw_stem, decoders=ALL_DECODERS, feature=FEATURE, method=METHOD,
                          durations=DURATIONS):
    """{decoder: {rmse_mean, rmse_err, cc_mean, cc_err}}, each an array over
    durations (NaN where a result is missing); errors are SEM across folds."""
    results = {}
    for decoder in decoders:
        stats_by_key = {k: [] for k in ("rmse_mean", "rmse_err", "cc_mean", "cc_err")}
        for minutes in durations:
            path = result_path(raw_stem, decoder, feature, method, minutes)
            if not os.path.exists(path):
                print(f"[missing] {decoder.upper()} @ {minutes} min: {path}")
                for values in stats_by_key.values():
                    values.append(np.nan)
                continue
            with h5py.File(path, "r") as f:
                rmse, cc = f["rmse_test_folds"][()], f["cc_test_folds"][()]
            stats_by_key["rmse_mean"].append(np.nanmean(rmse))
            stats_by_key["rmse_err"].append(stats.sem(rmse, nan_policy="omit"))
            stats_by_key["cc_mean"].append(np.nanmean(cc))
            stats_by_key["cc_err"].append(stats.sem(cc, nan_policy="omit"))
        results[decoder] = {k: np.array(v) for k, v in stats_by_key.items()}
    return results


def plot_full_comparison(raw_stem, results, feature=FEATURE, method=METHOD, save=True):
    """Bar chart of mean RMSE and CC (+/- SEM across folds) per decoder."""
    decoders = list(results)
    x = np.arange(len(decoders))
    labels = [d.upper() for d in decoders]
    colors = [DECODER_COLORS[d] for d in decoders]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, (ylabel, metric) in enumerate((("Average RMSE", 0), ("Average CC", 1))):
        means = [np.mean(results[d][metric]) for d in decoders]
        errs = [stats.sem(results[d][metric]) for d in decoders]
        ax[i].bar(x, means, yerr=errs, width=0.5, capsize=5, edgecolor="k", color=colors)
        customize_plot(ax[i], xlabel="Decoder", ylabel=ylabel, fontsize=13,
                       xticks=x, xticklabels=labels)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(subject_dirs(raw_stem)["figures"],
                                 f"{raw_stem}_{feature}_{method}_decoder.png"), bbox_inches="tight")

    rmse_means = [np.mean(results[d][0]) for d in decoders]
    print(f"Best decoder by RMSE: {decoders[int(np.argmin(rmse_means))].upper()} "
          f"({min(rmse_means):.2f})")
    return fig


def plot_duration_comparison(raw_stem, duration_results, full_results=None, durations=DURATIONS,
                             feature=FEATURE, method=METHOD, save=True):
    """RMSE and CC vs. training duration; dashed lines mark full-data results."""
    decoders = list(duration_results)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for decoder in decoders:
        r, color = duration_results[decoder], DECODER_COLORS[decoder]
        for i, metric in enumerate(("rmse", "cc")):
            ax[i].errorbar(durations, r[f"{metric}_mean"], yerr=r[f"{metric}_err"], fmt="-o",
                           capsize=4, color=color, markeredgecolor="k", label=decoder.upper())
            if full_results and decoder in full_results:
                ax[i].axhline(np.mean(full_results[decoder][i]), color=color, linestyle="--",
                              alpha=0.6)

    for i, ylabel in enumerate(("Average RMSE", "Average CC")):
        customize_plot(ax[i], xlabel="Training data (min)", ylabel=ylabel, fontsize=13,
                       xticks=list(durations), xticklabels=[str(d) for d in durations])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(decoders), frameon=False, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(subject_dirs(raw_stem)["figures"],
                                 f"{raw_stem}_{feature}_{method}_decoder_vs_duration.png"),
                    bbox_inches="tight")
    return fig


# ============================================================================
# Drivers
# ============================================================================

def _evaluate_and_plot(raw_stem, dataset_filepath, feature, overwrite, wf_tap,
                       n_train_override=None, gap_samples=None):
    split = dict(n_train_override=n_train_override, gap_samples=gap_samples)
    eval_wf(raw_stem, dataset_filepath, wf_tap=wf_tap, feature=feature, overwrite=overwrite, **split)
    eval_kf(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite, **split)
    write_dl_decoder_configs(raw_stem, feature=feature, overwrite=overwrite)
    eval_dl_decoders(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite, **split)

    full_results = load_full_results(raw_stem, feature=feature)
    duration_results = load_duration_results(raw_stem, feature=feature)
    plot_full_comparison(raw_stem, full_results, feature=feature)
    plot_duration_comparison(raw_stem, duration_results, full_results=full_results, feature=feature)


def process_session(raw_stem=RAW_STEM, feature=FEATURE, overwrite=False):
    """Full pipeline for a continuous (.mat) session.

    Outputs are namespaced by raw_stem and feature, so different sessions can
    run concurrently (cap each process's thread count, e.g. OMP_NUM_THREADS,
    before TensorFlow is imported)."""
    download_raw_data(raw_stem, overwrite=overwrite)
    processed_filepath = process_raw_data(raw_stem, feature=feature, overwrite=overwrite)
    dataset_filepath = build_ann_dataset(raw_stem, processed_filepath, feature=feature,
                                         overwrite=overwrite)
    snn_dataset_filepath, _ = build_snn_dataset(raw_stem, processed_filepath, feature=feature,
                                                overwrite=overwrite)
    build_combined_snn_dataset(raw_stem, snn_dataset_filepath, feature=feature, group_size=8,
                               overwrite=overwrite)
    _evaluate_and_plot(raw_stem, dataset_filepath, feature, overwrite, wf_tap=WF_TAP)
    print(f"\nDone: {raw_stem} ({feature})")


def process_nwb_session(raw_stem, dataset_filepath, feature=FEATURE, overwrite=False, gap_samples=None):
    """Decoder evaluation for a trial-structured NWB session.

    dataset_filepath is the {session}_binning.h5 written by
    nwb_conversion/run_nwb_pipeline.sh. Its rows are concatenated trials, so
    the train/test boundary is read from its n_train attribute (a trial
    boundary, shared with the SNN dataset) rather than computed.

    gap_samples=None keeps the eval scripts' default purge gap: the final
    split already falls between trials, but the CV folds split rows at
    arbitrary points, often inside a trial, where neighbouring rows share
    most of their window. The gap costs the final model only the last
    ~140 rows of its last training trial."""
    with h5py.File(dataset_filepath, "r") as f:
        n_train = f.attrs.get("n_train")
    if n_train is None:
        raise ValueError(f"{dataset_filepath} has no 'n_train' attribute; expected the output of "
                         f"nwb_conversion/combine_trial_windows_to_ann_h5.py. Use process_session() "
                         f"for continuous sessions.")
    print(f"Train/test boundary from {dataset_filepath}: n_train={int(n_train)}, "
          f"gap_samples={'default' if gap_samples is None else gap_samples}")
    _evaluate_and_plot(raw_stem, dataset_filepath, feature, overwrite, wf_tap=WF_TAP_HKM,
                       n_train_override=int(n_train), gap_samples=gap_samples)
    print(f"\nDone: {raw_stem} ({feature}) [NWB/trial-structured]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_stem", type=str, default=RAW_STEM, help="Session identifier")
    parser.add_argument("--feature", type=str, default=FEATURE, choices=["mua", "sua"])
    parser.add_argument("--overwrite", action="store_true", help="Recompute every stage")
    parser.add_argument("--nwb_dataset_filepath", type=str, default=None,
                        help="Run process_nwb_session() on this {session}_binning.h5 instead of "
                             "the full .mat pipeline")
    parser.add_argument("--gap_samples", type=int, default=None,
                        help="Purge gap for --nwb_dataset_filepath runs (default: the eval "
                             "scripts' default, see bmi/evaluation.py)")
    args = parser.parse_args()
    if args.nwb_dataset_filepath:
        process_nwb_session(args.raw_stem, args.nwb_dataset_filepath, feature=args.feature,
                            overwrite=args.overwrite, gap_samples=args.gap_samples)
    else:
        process_session(raw_stem=args.raw_stem, feature=args.feature, overwrite=args.overwrite)
