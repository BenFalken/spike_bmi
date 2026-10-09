#!/usr/bin/env python
"""
Single-subject BMI decoding pipeline: download raw data, build ANN/SNN
datasets, evaluate KF/WF/LSTM/QRNN decoders (both full-data and
duration-swept), and produce quick-look comparison figures.

THIS REVISION removes every optimization/search stage in favor of
hardcoded parameters, to stop spending time on per-session searches:

  - WF: no more opt_wf_decoder.py tap sweep. WF_TAP is now a hardcoded
    constant, set to the paper's own reported tap count (15).
  - LSTM/QRNN: no more opt_dl_decoder.py hyperparameter search.
    write_dl_decoder_configs() writes each decoder's config JSON directly
    from DL_HYPERPARAMS/DL_FIXED below -- no subprocess call, no search.
    Values are the paper's own reported hyperparameters (see
    DL_HYPERPARAMS's comment for exactly which table rows map onto this
    codebase's decoder set, and which don't).
  - MLP has been dropped from this pipeline entirely (not needed for the
    downstream analyses) -- DL_DECODERS/ALL_DECODERS no longer include it.

CONFIRMED against eval_dl_decoders.py's actual source (previously this was
an inferred guess): write_dl_decoder_configs()'s JSON needs exactly
{timesteps, n_layers, units, batch_size, learning_rate, dropout,
optimizer, epochs} -- eval_dl_decoders.py's main() does `config =
json.load(f)` directly when --config_filepath is set, then separately adds
input_dim/output_dim/window_size/loss/metric itself from the data/CLI args
-- those five do NOT belong in the written file, and correctly aren't
written here.

--feature is now a real CLI argument (default "mua"), not just a constant
-- see feature_dirs() below. FEATURE/DIRS previously baked "mua" into
three path entries at MODULE IMPORT time, which would have made a
--feature flag silently do nothing; feature_dirs(feature) now computes
those three paths fresh per call instead.

WINDOWING REGIME: the ANN decoders and the SNN use DELIBERATELY different
windowing, both built from one shared WDW_TIME -- see the "Windowing"
block in the Configuration section. ANN gets dense, near-total-overlap
4ms-step windows (matches the paper's own convention); SNN gets
non-overlapping WDW_TIME trials. This requires make_snn_dataset.py's
non_overlapping fix (--ol_time 0.0 previously still produced a 1-sample
overlap) and gap-aware eval_*.py scripts.

STILL NEEDED, out of scope for this file (would require editing OTHER
scripts, not this orchestration layer):
  - Aligning every train/test split to a 256ms (65-sample) boundary, so
    ANN and SNN datasets for the same session are guaranteed to begin
    their test split at the same real moment. This script only PASSES
    --test_frac through to make_dataset.py / eval_wf_decoder.py /
    eval_kf_decoder.py / eval_dl_decoders.py / export_snn_pkl.py -- it has
    no windowing/splitting logic of its own to edit. Each of those scripts
    needs the same compute_aligned_split()-style fix already applied to
    eval_all_decoders.py (round the train boundary to the nearest 65-
    sample multiple, computed from the RAW session length so every script
    derives the identical boundary) before this is actually consistent
    end to end.
  - bmi.decoders.KalmanDecoder.fit()'s regularized branches reference
    self.alpha_reg, which is never set (only self.reg_alpha is) --
    requesting a regularized KF fit raises AttributeError until that
    one-line typo is fixed in decoders.py. No --reg_type is passed for KF
    below because of this.
"""

import json
import os
import subprocess
import sys
import re

import h5py
import numpy as np
import requests
from scipy import stats
import matplotlib.pyplot as plt

# bmi/ is a shared package living at the project root, not a sibling of
# this file (which itself lives one level down, in preprocessing_
# training/) -- without this, `from bmi.utils import ...` below only
# resolves by accident, whenever cwd or sys.path[0] happens to already
# include the project root; confirmed directly this fails otherwise
# (ModuleNotFoundError) once this file is actually run from inside its
# own subdirectory, the same class of bug already fixed for every other
# subprocess call in this file via SCRIPT_DIR.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bmi.utils import customize_plot, legend_plot

# Force line-buffered stdout for THIS process's own print() calls -- see
# run_cmd()'s docstring below for the full explanation. Without this,
# this script's own prints can sit unflushed for the entire remainder of
# a long-running stage under SLURM (stdout redirected to a file, not a
# TTY, triggers Python's default full-buffering behavior), making
# genuine (if slow) progress look identical to a hang in the log --
# confirmed to be at least part of a real "the pipeline seems to freeze
# after build_snn_dataset" report.
sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# Storage roots -- overridable via environment variable, so there is
# exactly ONE place to change if the underlying storage location ever
# needs to move again (as it just did: the original "../data/bfalkenb/
# data" location lives adjacent to the home-directory quota, which was
# found to be SOFT_EXCEEDED and was silently truncating large raw-data
# transfers into it -- see the actual incident this was diagnosed from).
#
# BMI_DATA_ROOT: raw/processed/dataset-cache storage -- large, per-
# session files that benefit from scratch's much larger quota and don't
# need long-term backup guarantees the way code does.
#
# BMI_PROJECT_ROOT: the project's own root directory (containing
# datasets/, checkpoints/, results/, params/, figures/) -- ABSOLUTE, not
# derived from cwd. This was previously an implicit, relative path
# ("datasets/...", "results/...", etc), which worked fine only as long as
# every script lived in one flat directory; now that scripts are split
# across preprocessing_training/, snn_training/, visualization/, and
# analysis/, the same relative path silently resolves to a DIFFERENT,
# wrong location depending on which of those directories you happen to
# invoke a script from (confirmed directly: a "datasets/" subdirectory
# was found nested inside preprocessing_training/ itself, rather than at
# the intended project root, for exactly this reason). Anchoring to an
# absolute root removes that dependency on cwd entirely.
#
# Defaults below match this project's actual, current locations -- override
# either with, e.g.:
#   export BMI_DATA_ROOT=/oscar/scratch/bfalkenb/data
#   export BMI_PROJECT_ROOT=/oscar/home/bfalkenb/spike_bmi_main
BMI_DATA_ROOT = os.environ.get("BMI_DATA_ROOT", "/users/bfalkenb/scratch/bfalkenb/data")
BMI_PROJECT_ROOT = os.environ.get("BMI_PROJECT_ROOT", "/users/bfalkenb/spike_bmi_main")

# The directory this file itself lives in -- process_data.py, make_dataset.py,
# and every other stage script run_cmd() invokes below are expected to be
# SIBLINGS of this file (both live in preprocessing_training/). Every
# run_cmd() call below joins this against the bare filename, rather than
# passing the bare filename alone, specifically because a bare relative
# filename resolves against the CALLING PROCESS's cwd at the moment
# subprocess.run() executes -- which is wherever `sbatch` (or python3)
# was invoked FROM, not wherever this file happens to sit on disk. Under
# SLURM specifically, that's whatever directory a job script's own
# working directory happens to be (often a separate sbatch_scripts/
# directory, one level up and over from here) -- confirmed directly as
# the actual cause of a real "can't open file .../sbatch_scripts/
# process_data.py" failure, not a hypothetical concern.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Configuration
# ============================================================================

RAW_STEM = "indy_20160407_02"              # example session; override per subject
FEATURE = "mua"                             # default --feature; "mua" or "sua"
METHOD = "binning"
TEST_FRAC = 0.1
TRAIN_DURATIONS = "1,2,3,4,5,6,7,8,9,10"   # minutes
DURATIONS = tuple(range(1, 11))

# CV fold guard-rail headroom -- see prior revision's note: the KF/WF/DL
# scripts' own --min_train_size default (0.5) exactly equals their
# --n_folds x --test_size defaults (5 x 0.1 = 0.5), a zero-margin design
# that trips as soon as any purge gap exists between train/test. 0.45
# restores real headroom without changing how much data is actually used.
MIN_TRAIN_SIZE = 0.45

# --- Windowing ---
# ANN: dense, 4ms-step windows (near-total overlap) -- matches the paper's
#      own convention.
# SNN: non-overlapping WDW_TIME-wide trials -- see make_snn_dataset.py's
#      --ol_time 0.0 non_overlapping fix.
# ANN_STEP_MS is what the ANN-side eval scripts' --step_ms/--wdw_time
# flags need (train-duration sample-count conversion, and their default
# purge-gap computation) -- it must be the ANN dataset's actual row
# spacing, not the SNN's.
WDW_TIME = 0.256
ANN_OL_TIME = 0.252     # step = WDW_TIME - ANN_OL_TIME = 0.004s = native 4ms
ANN_STEP_MS = 4.0
SNN_OL_TIME = 0.0       # non-overlapping -- see make_snn_dataset.py's non_overlapping fix

# WF: paper's own tap count, hardcoded -- see module docstring (no more
# opt_wf_decoder.py sweep).
WF_TAP = 15
# HKM's own 192 channels (vs bmi's 96) make WF_TAP=15's flattened
# feature dimension (channels * timesteps) large enough to have
# genuinely failed in practice -- confirmed directly, not hypothetical
# (OOM even at 128GB, multi-hour runtime). This is a MEMORY/COMPUTE
# constraint specific to hkm's channel count, not a change to the
# paper's own empirically-set WF_TAP=15 for bmi -- process_session()
# (bmi) keeps using the default above; only process_nwb_session() (hkm)
# overrides it with this. test_all_decoders.py already reads each
# saved model's own 'timesteps' back from its config json (confirmed
# directly) rather than assuming a fixed value, so a bmi/hkm split here
# needs no changes anywhere downstream.
WF_TAP_HKM = 8
WF_REG_TYPE = "l2"
WF_REG_ALPHA = 1.0      # still a placeholder, not cross-validated -- optimize_wf_decoder.py
                          # was never integrated into this pipeline even before this revision

DL_DECODERS = ("lstm", "qrnn")     # MLP dropped entirely -- not needed for the ultimate analyses
# SNN is trained/evaluated by a separate pipeline (checkpoints live under
# checkpoints/bmi/, built outside this script) -- not included here. See
# eval_all_decoders.py for the comparison that does include it.
ALL_DECODERS = ("kf", "wf", "lstm", "qrnn")

# Hardcoded from the paper's own reported hyperparameters (see the
# uploaded hyperparameter table). Only LSTM and QRNN map directly onto
# this codebase's decoder set -- the paper's SRNN and GRU rows have no
# equivalent here and are ignored; the paper has no MLP row at all, which
# is moot now that MLP has been dropped. Applied UNIFORMLY regardless of
# --feature (mua vs sua): the paper's table is reported for its own "ESA
# Signal" input, not ours, but matching those values exactly -- rather
# than re-deriving feature-specific ones via a search -- is the deliberate
# tradeoff being made here to stop spending time on optimization runs.
DL_HYPERPARAMS = {
    "lstm": {"units": 200, "epochs": 5, "batch_size": 32, "dropout": 0.5, "learning_rate": 0.004},
    "qrnn": {"units": 400, "epochs": 2, "batch_size": 64, "dropout": 0.5, "learning_rate": 0.0015},
}
# The paper's table doesn't cover timesteps/n_layers/optimizer at all --
# those come from eval_dl_decoders.py's own established CLI defaults
# (--timesteps 5, --n_layers 1, --optimizer Adam), confirmed against real
# reference config JSONs from prior optimized runs, which use these exact
# same three values. (An earlier revision of this file had timesteps=2 and
# optimizer="rmsprop" here -- neither was ever actually sourced from the
# paper table or verified against eval_dl_decoders.py; both were wrong.)
DL_FIXED = {"timesteps": 5, "n_layers": 1, "optimizer": "Adam"}

# Kept consistent with every other figure in this project.
DECODER_COLORS = {
    "kf": "purple", "wf": "goldenrod", "mlp": "crimson",
    "lstm": "darkorange", "qrnn": "seagreen", "snn": "royalblue",
}

# EXPERIMENT is DERIVED per-call from raw_stem's own format, not a fixed
# module-level constant -- this module handles BOTH the original,
# .mat-based subjects (process_session(), raw_stem like
# 'indy_20160407_02' -- experiment "bmi") AND NWB-derived subjects
# (process_nwb_session(), raw_stem like
# 'sub-Nitschke_ses-20090812_behavior+ecephys' -- experiment "hkm"). A
# fixed constant would have been silently WRONG for whichever of the two
# it didn't match -- e.g. process_nwb_session()'s own results landing
# under results/bmi/... instead of results/hkm/..., discovered by
# tracing through every caller of experiment_and_subject_from_raw_stem()
# rather than
# assumed safe from process_session()'s call sites alone.
_NWB_RAW_STEM_RE = re.compile(r"^sub-([A-Za-z]+)_ses-")


def experiment_and_subject_from_raw_stem(raw_stem):
    """Derives (experiment, subject) from raw_stem's own format --
    confirmed against both real conventions actually used in this
    project:
      'indy_20160407_02'                        -> ('bmi', 'indy')
      'sub-Nitschke_ses-20090812_behavior+ecephys' -> ('hkm', 'nitschke')
    A separate, explicit --experiment/--subject flag would risk silently
    drifting out of sync with the raw_stem actually being processed;
    deriving both directly from the same string every other path is
    built from cannot drift."""
    m = _NWB_RAW_STEM_RE.match(raw_stem)
    if m:
        return "hkm", m.group(1).lower()
    subject = raw_stem.split("_")[0]
    if not subject:
        raise ValueError(f"could not derive an experiment/subject from raw_stem={raw_stem!r} -- "
                          f"expected either the NWB 'sub-{{Subject}}_ses-...' convention or the "
                          f"original '{{subject}}_{{date}}_{{index}}' convention.")
    return "bmi", subject


# Feature-INDEPENDENT directories only -- see feature_dirs() below for the
# three that must be namespaced per --feature (mua/sua), and subject_dirs()
# for the three that must be namespaced per experiment/subject (results/params/figures).
DIRS = {}


def feature_dirs(raw_stem, feature):
    """Directories that must be namespaced per EXPERIMENT/SUBJECT (derived
    from raw_stem) AND per --feature (mua/sua), so runs for different
    subjects or features never collide, and never land in a shared, flat
    directory that would need raw_stem alone to disambiguate. Computed
    fresh per call (not baked into a module-level dict at import time) --
    that's exactly what would have made --feature (or a different
    subject) silently do nothing."""
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    dirs = {
        "processed": os.path.join(BMI_DATA_ROOT, "processed", experiment, subject, feature),
        "dataset": os.path.join(BMI_DATA_ROOT, "dataset", experiment, subject, feature),
        "snn_pkl_root": os.path.join(BMI_DATA_ROOT, "snn_datasets", experiment, subject, feature),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def subject_dirs(raw_stem):
    """Results/params/figures directories, namespaced per EXPERIMENT/
    SUBJECT (derived from raw_stem) -- these used to be flat, subject-
    independent module-level constants (DIRS["results_decoder"], etc),
    which meant every subject's own results/configs/figures landed in
    the SAME shared directory, disambiguated only by raw_stem being
    baked into each individual filename. That's enough to avoid
    filename COLLISIONS, but not enough to keep results genuinely
    separated for browsing, caching, or cleanup -- a real, explicit
    per-experiment/subject directory does that instead."""
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    dirs = {
        "results_decoder": os.path.join(BMI_DATA_ROOT, "results", experiment, subject, "decoder"),
        "params": os.path.join(BMI_DATA_ROOT, "params", experiment, subject),
        "figures": os.path.join(BMI_DATA_ROOT, "figures", experiment, subject),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def raw_dir(raw_stem):
    """Raw-data directory, namespaced per EXPERIMENT/SUBJECT -- matches
    this project's own actual raw data layout
    (data/raw/{experiment}/{subject}/{raw_stem}.mat), confirmed directly
    against the real directory listing rather than assumed. Only ever
    actually used for the 'bmi' experiment's own .mat files (see
    download_raw_data()/process_raw_data() -- NWB files arrive via a
    separate route, run_nwb_pipeline.sh, not this function)."""
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    d = os.path.join(BMI_DATA_ROOT, "raw", experiment, subject)
    os.makedirs(d, exist_ok=True)
    return d


def model_cache_dir(raw_stem, feature=FEATURE):
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    path = os.path.join(BMI_DATA_ROOT, "results", "model_cache", experiment, subject, raw_stem, feature)
    os.makedirs(path, exist_ok=True)
    return path


# Whichever interpreter is running THIS script, used for every subprocess
# call below instead of a literal "python" -- sidesteps the "python vs
# python3" question entirely.
PYTHON = sys.executable


def run_cmd(args, description):
    """Run a CLI stage as a subprocess. Raises on failure so a broken
    stage doesn't silently continue into the next one.

    Inserts "-u" (unbuffered stdout/stderr) right after the interpreter --
    same buffering issue as this file's own sys.stdout.reconfigure() call
    at the top, but for the SUBPROCESS's own print() calls this time.
    Without it, a subprocess that's genuinely still working (e.g.
    export_snn_pkl.py writing thousands of individual small .pkl files,
    with no progress output between its first and last print) can sit
    silent in the log for the entire remainder of a long stage, making it
    indistinguishable from a real hang."""
    print(f"\n=== {description} ===")
    print(" ".join(args))
    subprocess.run([args[0], "-u"] + args[1:], check=True)


def _skip_if_exists(path, overwrite, description):
    if (not overwrite) and os.path.exists(path):
        print(f"[skip] {description}: {path} already exists")
        return True
    return False


# ============================================================================
# Stage 1: data acquisition and preprocessing
# ============================================================================

def download_raw_data(raw_stem, overwrite=False):
    """NOTE: inferred, not verified -- the actual download cell wasn't
    among the source shared for this pipeline's original consolidation."""
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
    if _skip_if_exists(processed_filepath, overwrite, "process_data"):
        return processed_filepath
    run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "process_data.py"),
             "--input_filepath", raw_filepath,
             "--output_filepath", processed_filepath],
            "process_data.py")
    return processed_filepath


def build_ann_dataset(raw_stem, processed_filepath, feature=FEATURE, method=METHOD, overwrite=False):
    """Dense, 4ms-step windows -- see WDW_TIME/ANN_OL_TIME in the config
    section."""
    dataset_filepath = os.path.join(feature_dirs(raw_stem, feature)["dataset"], f"{raw_stem}_{method}.h5")
    if _skip_if_exists(dataset_filepath, overwrite, "make_dataset"):
        return dataset_filepath
    run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "make_dataset.py"),
             "--input_filepath", processed_filepath,
             "--output_filepath", dataset_filepath,
             "--method", method,
             "--wdw_time", str(WDW_TIME),
             "--ol_time", str(ANN_OL_TIME)],
            "make_dataset.py")
    return dataset_filepath


def build_snn_dataset(raw_stem, processed_filepath, feature=FEATURE, overwrite=False):
    """Non-overlapping WDW_TIME-wide trials (SNN_OL_TIME=0.0). Requires
    the make_snn_dataset.py fix that makes --ol_time 0.0 actually produce
    zero shared raw samples between consecutive windows."""
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
             "--methods", "binning"],
            "check_datasets.py (independent ANN + SNN validation)")

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
    """Groups group_size consecutive base windows into longer trials
    (see combine_snn_dataset.py's own module docstring for the full
    motivation) -- reads the SAME snn_dataset_filepath build_snn_dataset()
    already produced (its own h5 intermediate, not the exported .pkl
    trials), so this only makes sense called AFTER build_snn_dataset()
    for the same raw_stem/feature, not standalone.

    discard_remainder defaults to True (--discard-train-remainder) --
    this project's own established requirement for batch_size>1 training
    on the grouped dataset (uniform trial length; the default DataLoader
    collate_fn cannot stack trials of different lengths, and DOES fail
    exactly this way -- confirmed directly from a real training run's
    own traceback -- if a shorter, trailing remainder trial is left in).
    Pass False only if you specifically want the remainder trial kept
    (e.g. batch_size=1 training, where uniform length isn't required).

    Output directory matches this project's established {feature}_
    {group_size}_group naming (e.g. mua_8_group), namespaced by
    experiment/subject exactly like feature_dirs()["snn_pkl_root"] --
    NOT reusing that dict directly since it's keyed by feature alone,
    with no group_size axis."""
    experiment, subject = experiment_and_subject_from_raw_stem(raw_stem)
    output_dir = os.path.join(BMI_DATA_ROOT, "datasets", experiment, subject,
                               f"{feature}_{group_size}_group")
    marker_path = os.path.join(output_dir, raw_stem, "train", "0.pkl")
    if _skip_if_exists(marker_path, overwrite, f"combine_snn_dataset (group_size={group_size})"):
        return output_dir

    combine_args = [PYTHON, os.path.join(SCRIPT_DIR, "combine_snn_dataset.py"),
             "--h5-path", snn_dataset_filepath,
             "--session-id", raw_stem,
             "--group-size", str(group_size),
             "--test-frac", str(test_frac),
             "--output-dir", output_dir]
    if discard_remainder:
        combine_args.append("--discard-train-remainder")
    run_cmd(combine_args, f"combine_snn_dataset.py (group_size={group_size})")
    return output_dir


# ============================================================================
# Stage 2: Wiener filter + Kalman filter eval (hardcoded WF_TAP, no sweep)
# ============================================================================

def eval_linear_decoder(decoder, raw_stem, dataset_filepath, feature=FEATURE, method=METHOD,
                         extra_args=(), overwrite=False, n_train_override=None, gap_samples=None):
    """Shared full-data + duration-swept eval for WF and KF, via
    eval_{decoder}_decoder.py: train_durations left empty reproduces the
    full-data-only behavior; --train_durations set additionally sweeps.

    n_train_override/gap_samples: see module docstring's "STILL NEEDED"
    note -- eval_wf_decoder.py/eval_kf_decoder.py now support both
    directly (mirroring the identical fix already applied to
    test_all_decoders.py's own run_session()). Both default to None here,
    meaning the CONTINUOUS-session data this pipeline was originally
    built for is completely unaffected -- these only matter for TRIAL-
    STRUCTURED data (e.g. the NWB/Jenkins-Nitschke conversion), where
    they're required for the split to be correct at all, not optional
    tuning. See process_nwb_session() below."""
    model_dir = model_cache_dir(raw_stem, feature)
    script = os.path.join(SCRIPT_DIR, f"eval_{decoder}_decoder.py")
    results_decoder_dir = subject_dirs(raw_stem)["results_decoder"]
    full_result = os.path.join(results_decoder_dir, f"{raw_stem}_{feature}_{method}_{decoder}.h5")
    common_args = ["--input_filepath", dataset_filepath,
                   "--output_filepath", full_result,
                   "--model_dir", model_dir,
                   "--test_frac", str(TEST_FRAC),
                   "--feature", feature,
                   "--wdw_time", str(WDW_TIME),
                   "--step_ms", str(ANN_STEP_MS),
                   "--min_train_size", str(MIN_TRAIN_SIZE),
                   *extra_args]
    if n_train_override is not None:
        common_args += ["--n_train_override", str(n_train_override)]
    if gap_samples is not None:
        common_args += ["--gap_samples", str(gap_samples)]

    # The h5 RESULTS file and the cached MODEL file are written by two
    # SEPARATE steps inside eval_{decoder}_decoder.py -- the h5 write
    # happens first, well before the later, separate model_dir pickling
    # step (see that script's own --model_dir block). An OOM/crash during
    # pickling can therefore leave a perfectly valid h5 file behind while
    # the model cache never gets written at all -- confirmed directly as
    # a real failure mode, not hypothetical. Checking full_result alone
    # would then silently skip re-running forever, since the h5 file by
    # itself looks "done." _needs_rerun() below requires BOTH to exist.
    def _needs_rerun(result_path, model_path, label):
        if overwrite:
            return True
        if not os.path.exists(result_path):
            return True
        if not os.path.exists(model_path):
            print(f"[incomplete] {label}: {result_path} exists, but its own cached model "
                  f"{model_path} does not -- likely an OOM/crash during model pickling, "
                  f"AFTER the h5 output was already written. Re-running.")
            return True
        print(f"[skip] {label}: {result_path} already exists")
        return False

    if _needs_rerun(full_result, os.path.join(model_dir, f"{decoder}_model.pkl"),
                     f"{decoder.upper()} full-data eval"):
        run_cmd([PYTHON, script, *common_args], f"{decoder.upper()} full-data eval")

    last_duration_marker = os.path.join(
        results_decoder_dir, f"{raw_stem}_{feature}_{method}_{decoder}_{DURATIONS[-1]}min.h5")
    last_duration_model = os.path.join(model_dir, f"{decoder}_{DURATIONS[-1]:g}min_model.pkl")
    if _needs_rerun(last_duration_marker, last_duration_model, f"{decoder.upper()} duration sweep"):
        run_cmd([PYTHON, script, *common_args, "--train_durations", TRAIN_DURATIONS],
                f"{decoder.upper()} duration sweep")
    return full_result


def eval_wf(raw_stem, dataset_filepath, wf_tap=WF_TAP, feature=FEATURE, method=METHOD, overwrite=False,
            n_train_override=None, gap_samples=None):
    """wf_tap defaults to the hardcoded WF_TAP (paper's own value) -- no
    more opt_wf_decoder.py sweep feeding this in."""
    return eval_linear_decoder(
        "wf", raw_stem, dataset_filepath, feature, method,
        extra_args=["--timesteps", str(wf_tap),
                    "--reg_type", WF_REG_TYPE, "--reg_alpha", str(WF_REG_ALPHA)],
        overwrite=overwrite, n_train_override=n_train_override, gap_samples=gap_samples)


def eval_kf(raw_stem, dataset_filepath, feature=FEATURE, method=METHOD, overwrite=False,
            n_train_override=None, gap_samples=None):
    """No --reg_type is passed: KalmanDecoder's regularized branches
    currently crash (self.alpha_reg bug) -- see module docstring. Do not
    enable KF regularization until that's fixed in bmi/decoders.py."""
    return eval_linear_decoder("kf", raw_stem, dataset_filepath, feature, method,
                                extra_args=(), overwrite=overwrite,
                                n_train_override=n_train_override, gap_samples=gap_samples)


# ============================================================================
# Stage 3: deep-learning decoders -- hardcoded config, then eval
# ============================================================================

def write_dl_decoder_configs(raw_stem, decoders=DL_DECODERS, feature=FEATURE, method=METHOD,
                              overwrite=False):
    """Writes each decoder's config JSON DIRECTLY from DL_HYPERPARAMS/
    DL_FIXED -- no opt_dl_decoder.py subprocess, no search. See module
    docstring for the important caveat: the exact JSON field names here
    are inferred, not confirmed against eval_dl_decoders.py's actual
    config-reading code.
    """
    params_dir = subject_dirs(raw_stem)["params"]
    for decoder in decoders:
        config_filepath = os.path.join(params_dir, f"{raw_stem}_{feature}_{method}_{decoder}.json")
        if _skip_if_exists(config_filepath, overwrite, f"{decoder.upper()} hardcoded config"):
            continue
        config = {**DL_FIXED, **DL_HYPERPARAMS[decoder]}
        with open(config_filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote hardcoded {decoder.upper()} config to {config_filepath}: {config}")


def eval_dl_decoders(raw_stem, dataset_filepath, decoders=DL_DECODERS, feature=FEATURE,
                      method=METHOD, overwrite=False, n_train_override=None, gap_samples=None):
    """Both full-data and duration-swept runs go through eval_dl_decoders.py.
    See eval_linear_decoder()'s docstring for n_train_override/gap_samples."""
    model_dir = model_cache_dir(raw_stem, feature)
    dirs = subject_dirs(raw_stem)
    for decoder in decoders:
        config_filepath = os.path.join(dirs["params"], f"{raw_stem}_{feature}_{method}_{decoder}.json")
        full_result = os.path.join(dirs["results_decoder"], f"{raw_stem}_{feature}_{method}_{decoder}.h5")
        common_args = ["--input_filepath", dataset_filepath,
                       "--output_filepath", full_result,
                       "--decoder", decoder,
                       "--config_filepath", config_filepath,
                       "--model_dir", model_dir,
                       "--test_frac", str(TEST_FRAC),
                       "--feature", feature,
                       "--wdw_time", str(WDW_TIME),
                       "--step_ms", str(ANN_STEP_MS),
                       "--min_train_size", str(MIN_TRAIN_SIZE)]
        if n_train_override is not None:
            common_args += ["--n_train_override", str(n_train_override)]
        if gap_samples is not None:
            common_args += ["--gap_samples", str(gap_samples)]

        if not _skip_if_exists(full_result, overwrite, f"{decoder.upper()} full-data eval"):
            run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "eval_dl_decoders.py"), *common_args],
                    f"{decoder.upper()} full-data eval")

        last_duration_marker = os.path.join(
            dirs["results_decoder"], f"{raw_stem}_{feature}_{method}_{decoder}_{DURATIONS[-1]}min.h5")
        if not _skip_if_exists(last_duration_marker, overwrite, f"{decoder.upper()} duration sweep"):
            run_cmd([PYTHON, os.path.join(SCRIPT_DIR, "eval_dl_decoders.py"), *common_args,
                     "--train_durations", TRAIN_DURATIONS],
                    f"{decoder.upper()} duration sweep")


# ============================================================================
# Stage 4: quick-look comparison figures (single subject only)
# ============================================================================
# For anything beyond a single-subject sanity check, use the multi-session
# tools built separately for this project: eval_all_decoders.py
# (per-session and --multi_session comparison, including SNN) and
# plot_decoder_comparison_4x2.py (the report-grade summary figure). SNN is
# intentionally excluded here -- it's trained/evaluated by a separate
# pipeline not part of this script.

def load_full_results(raw_stem, decoders=ALL_DECODERS, feature=FEATURE, method=METHOD):
    results = {}
    results_decoder_dir = subject_dirs(raw_stem)["results_decoder"]
    for decoder in decoders:
        path = os.path.join(results_decoder_dir, f"{raw_stem}_{feature}_{method}_{decoder}.h5")
        if not os.path.exists(path):
            print(f"[skip] {decoder.upper()}: no full-data result at {path}")
            continue
        with h5py.File(path, "r") as f:
            results[decoder] = (f["rmse_test_folds"][()], f["cc_test_folds"][()])
    return results


def load_duration_results(raw_stem, decoders=ALL_DECODERS, feature=FEATURE, method=METHOD,
                           durations=DURATIONS):
    results = {}
    results_decoder_dir = subject_dirs(raw_stem)["results_decoder"]
    for decoder in decoders:
        rmse_means, rmse_errs, cc_means, cc_errs = [], [], [], []
        for minutes in durations:
            path = os.path.join(
                results_decoder_dir, f"{raw_stem}_{feature}_{method}_{decoder}_{minutes:g}min.h5")
            if not os.path.exists(path):
                print(f"[missing] {decoder.upper()} @ {minutes} min: {path}")
                rmse_means.append(np.nan); rmse_errs.append(np.nan)
                cc_means.append(np.nan); cc_errs.append(np.nan)
                continue
            with h5py.File(path, "r") as f:
                rmse_test, cc_test = f["rmse_test_folds"][()], f["cc_test_folds"][()]
            rmse_means.append(np.nanmean(rmse_test))
            rmse_errs.append(stats.sem(rmse_test, nan_policy="omit"))
            cc_means.append(np.nanmean(cc_test))
            cc_errs.append(stats.sem(cc_test, nan_policy="omit"))
        results[decoder] = {
            "rmse_mean": np.array(rmse_means), "rmse_err": np.array(rmse_errs),
            "cc_mean": np.array(cc_means), "cc_err": np.array(cc_errs),
        }
    return results


def plot_full_comparison(raw_stem, results, feature=FEATURE, method=METHOD, save=True):
    decoders = list(results.keys())
    rmse_mean = [np.mean(results[d][0]) for d in decoders]
    rmse_err = [stats.sem(results[d][0]) for d in decoders]
    cc_mean = [np.mean(results[d][1]) for d in decoders]
    cc_err = [stats.sem(results[d][1]) for d in decoders]
    x = np.arange(len(decoders))
    bar_colors = [DECODER_COLORS[d] for d in decoders]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].bar(x, rmse_mean, yerr=rmse_err, width=0.5, capsize=5, edgecolor="k", color=bar_colors)
    customize_plot(ax[0], xlabel="Decoder", ylabel="Average RMSE", fontsize=13,
                   xticks=x, xticklabels=[d.upper() for d in decoders], xlim=None, ylim=None, rotation=0)
    ax[1].bar(x, cc_mean, yerr=cc_err, width=0.5, capsize=5, edgecolor="k", color=bar_colors)
    customize_plot(ax[1], xlabel="Decoder", ylabel="Average CC", fontsize=13,
                   xticks=x, xticklabels=[d.upper() for d in decoders], xlim=None, ylim=None, rotation=0)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(subject_dirs(raw_stem)["figures"], f"{raw_stem}_{feature}_{method}_decoder.png"),
                    bbox_inches="tight")

    best = decoders[int(np.argmin(rmse_mean))]
    print(f"Best decoder by RMSE: {best.upper()} ({min(rmse_mean):.2f})")
    return fig


def plot_duration_comparison(raw_stem, duration_results, full_results=None, durations=DURATIONS,
                              feature=FEATURE, method=METHOD, save=True):
    decoders = list(duration_results.keys())
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for decoder in decoders:
        r, color = duration_results[decoder], DECODER_COLORS[decoder]
        ax[0].errorbar(durations, r["rmse_mean"], yerr=r["rmse_err"], fmt="-o", capsize=4,
                       color=color, markeredgecolor="k", label=decoder.upper())
        ax[1].errorbar(durations, r["cc_mean"], yerr=r["cc_err"], fmt="-o", capsize=4,
                       color=color, markeredgecolor="k", label=decoder.upper())
        if full_results and decoder in full_results:
            ax[0].axhline(np.mean(full_results[decoder][0]), color=color, linestyle="--", alpha=0.6)
            ax[1].axhline(np.mean(full_results[decoder][1]), color=color, linestyle="--", alpha=0.6)

    customize_plot(ax[0], xlabel="Training data (min)", ylabel="Average RMSE", fontsize=13,
                   xticks=list(durations), xticklabels=[str(d) for d in durations], xlim=None, ylim=None, rotation=0)
    customize_plot(ax[1], xlabel="Training data (min)", ylabel="Average CC", fontsize=13,
                   xticks=list(durations), xticklabels=[str(d) for d in durations], xlim=None, ylim=None, rotation=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(decoders), frameon=False, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(subject_dirs(raw_stem)["figures"], f"{raw_stem}_{feature}_{method}_decoder_vs_duration.png"),
                    bbox_inches="tight")
    return fig


# ============================================================================
# Driver
# ============================================================================

def process_session(raw_stem=RAW_STEM, feature=FEATURE, overwrite=False):
    """Full single-subject pipeline. Every side effect is file-based and
    session-namespaced under raw_stem/feature, so this is safe to call for
    different subjects (or the same subject's different features)
    concurrently -- as long as each caller caps TensorFlow/PyTorch's
    thread count first (each will otherwise try to claim every core; not
    this function's job to set that, since it should be set once per
    worker process, before any TF/PyTorch import happens)."""
    download_raw_data(raw_stem, overwrite=overwrite)
    processed_filepath = process_raw_data(raw_stem, feature=feature, overwrite=overwrite)
    dataset_filepath = build_ann_dataset(raw_stem, processed_filepath, feature=feature, overwrite=overwrite)
    snn_dataset_filepath, _ = build_snn_dataset(raw_stem, processed_filepath, feature=feature, overwrite=overwrite)
    build_combined_snn_dataset(raw_stem, snn_dataset_filepath, feature=feature,
                                group_size=8, overwrite=overwrite)

    eval_wf(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite)
    eval_kf(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite)

    write_dl_decoder_configs(raw_stem, feature=feature, overwrite=overwrite)
    eval_dl_decoders(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite)

    full_results = load_full_results(raw_stem, feature=feature)
    duration_results = load_duration_results(raw_stem, feature=feature)
    plot_full_comparison(raw_stem, full_results, feature=feature)
    plot_duration_comparison(raw_stem, duration_results, full_results=full_results, feature=feature)

    print(f"\nDone: {raw_stem} ({feature})")


def process_nwb_session(raw_stem, dataset_filepath, feature=FEATURE, overwrite=False, gap_samples=0):
    """Counterpart to process_session() for NWB/Jenkins-Nitschke-derived
    data (run_nwb_pipeline.sh's output) -- SKIPS download_raw_data /
    process_raw_data / build_ann_dataset / build_snn_dataset entirely,
    since that data doesn't come from a .mat-style raw file at all and
    run_nwb_pipeline.sh already produced dataset_filepath (the combined
    {session}_binning.h5) and the SNN train/test/*.pkl directory (trained
    separately via train_bmi.py, same as this pipeline's own SNN --
    see process_session()'s own comment on that).

    n_train_override is read DIRECTLY from dataset_filepath's own 'n_train'
    attr (written by combine_trial_windows_to_ann_h5.py) and passed
    through to every decoder eval -- this is exactly the "STILL NEEDED"
    gap this module's own docstring named before this data existed to
    need it. gap_samples defaults to 0 here (not None, unlike
    process_session()'s own decoder calls) -- trial boundaries already
    provide a genuine gap (no shared raw samples between different
    trials), so the usual purge-gap protection is unnecessary and would
    only discard real, valid rows of the last training trial for no
    benefit; pass a nonzero value explicitly if you have a specific
    reason to want extra purging anyway.
    """
    with h5py.File(dataset_filepath, "r") as f:
        n_train_override = f.attrs.get("n_train")
    if n_train_override is None:
        raise ValueError(
            f"{dataset_filepath} has no 'n_train' attr -- this doesn't look like "
            f"combine_trial_windows_to_ann_h5.py's output. process_nwb_session() only "
            f"makes sense for trial-structured data with a pre-computed, trial-aware "
            f"split boundary; for normal continuous-session data, use process_session() "
            f"instead, which lets each decoder script derive its own boundary.")
    n_train_override = int(n_train_override)
    print(f"Read n_train={n_train_override} from {dataset_filepath}'s own attr "
          f"(trial-aware boundary, gap_samples={gap_samples})")

    eval_wf(raw_stem, dataset_filepath, wf_tap=WF_TAP_HKM, feature=feature, overwrite=overwrite, n_train_override=n_train_override, gap_samples=gap_samples)
    eval_kf(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite, n_train_override=n_train_override, gap_samples=gap_samples)

    write_dl_decoder_configs(raw_stem, feature=feature, overwrite=overwrite)
    eval_dl_decoders(raw_stem, dataset_filepath, feature=feature, overwrite=overwrite,
                      n_train_override=n_train_override, gap_samples=gap_samples)

    full_results = load_full_results(raw_stem, feature=feature)
    duration_results = load_duration_results(raw_stem, feature=feature)
    plot_full_comparison(raw_stem, full_results, feature=feature)
    plot_duration_comparison(raw_stem, duration_results, full_results=full_results, feature=feature)

    print(f"\nDone: {raw_stem} ({feature}) [NWB/trial-structured]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_stem", type=str, default=RAW_STEM)
    parser.add_argument("--feature", type=str, default=FEATURE, choices=["mua", "sua"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nwb_dataset_filepath", type=str, default=None,
                         help="If set, run process_nwb_session() instead of the normal "
                              "process_session() -- points directly at run_nwb_pipeline.sh's "
                              "combined {session}_binning.h5 output (which carries its own "
                              "trial-aware 'n_train' attr) rather than downloading/processing "
                              "a raw .mat-style session. --raw_stem still applies, for result/"
                              "model-cache namespacing.")
    parser.add_argument("--gap_samples", type=int, default=0,
                         help="Only used with --nwb_dataset_filepath -- see "
                              "process_nwb_session()'s own docstring for why this defaults to "
                              "0 here specifically, unlike the normal pipeline.")
    args = parser.parse_args()
    if args.nwb_dataset_filepath:
        process_nwb_session(args.raw_stem, args.nwb_dataset_filepath, feature=args.feature,
                             overwrite=args.overwrite, gap_samples=args.gap_samples)
    else:
        process_session(raw_stem=args.raw_stem, feature=args.feature, overwrite=args.overwrite)
