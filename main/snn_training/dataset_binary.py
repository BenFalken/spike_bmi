"""
BMI Dataset Module
Dataset and DataLoader for per-trial .pt files produced by bmi_snn_pipeline.ipynb.
Each file contains 'spikes', 'position', 'markers', 'marker_labels', and 'trial_idx'.
"""
import json
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import pickle as pkl

# Velocity scaling constants -- LEGACY, Indy-specific values, kept
# EXACTLY as they were (name and value both) for backward compatibility
# with any other file that still imports V_LO/V_HI/V_MARGIN directly
# (the module docstring names make_large_snn_dataset.py specifically) --
# NOT verified against that file's actual source, so these are left
# untouched rather than risk silently breaking an import elsewhere.
#
# CustomDataset itself no longer uses these -- see
# get_velocity_scalers() below, which looks up each experiment/subject's
# OWN bounds (from compute_velocity_scalers.py's output) instead of
# using one hardcoded, Indy-only value for every subject. Any file
# still importing the bare constants below is, by definition, still
# Indy-only; migrate it to get_velocity_scalers("bmi", "indy") (which
# returns this exact same value, sourced from the same JSON, so the
# numbers are guaranteed consistent either way) when convenient.
V_LO, V_HI, V_MARGIN = -280.56, 316.54, 0.05

# --- Per-experiment/subject velocity scalers ---------------------------
# Path to the JSON produced by compute_velocity_scalers.py, mapping
# {experiment: {subject: {v_lo, v_hi, margin, ...}}}. Overridable via
# env var, matching this project's established BMI_DATA_ROOT/
# BMI_PROJECT_ROOT pattern -- one place to change if the file ever moves.
BMI_VELOCITY_SCALERS_PATH = os.environ.get(
    "BMI_VELOCITY_SCALERS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_scalers.json"),
)

_velocity_scalers_cache = None


def _load_velocity_scalers():
    """Loads and caches BMI_VELOCITY_SCALERS_PATH's JSON -- read from
    disk once per process, not once per __getitem__ call (this is called
    from inside CustomDataset.__init__, not __getitem__, for exactly
    that reason; DataLoader workers each re-import this module once, so
    each worker process still only pays this cost once, not per-sample)."""
    global _velocity_scalers_cache
    if _velocity_scalers_cache is None:
        if not os.path.isfile(BMI_VELOCITY_SCALERS_PATH):
            raise FileNotFoundError(
                f"BMI_VELOCITY_SCALERS_PATH={BMI_VELOCITY_SCALERS_PATH!r} does not exist -- "
                f"run compute_velocity_scalers.py first, or set BMI_VELOCITY_SCALERS_PATH "
                f"to point at an existing velocity_scalers.json.")
        with open(BMI_VELOCITY_SCALERS_PATH, "r") as f:
            _velocity_scalers_cache = json.load(f)
    return _velocity_scalers_cache


def get_velocity_scalers(experiment: str, subject: str) -> Tuple[float, float, float]:
    """Returns (v_lo, v_hi, margin) for one experiment/subject, from
    compute_velocity_scalers.py's own output -- the single source of
    truth for per-subject scaling, replacing the old single, Indy-only
    V_LO/V_HI/V_MARGIN constants above."""
    scalers = _load_velocity_scalers()
    if experiment not in scalers or subject not in scalers[experiment]:
        available = {exp: list(subs.keys()) for exp, subs in scalers.items()}
        raise KeyError(
            f"No velocity scalers found for experiment={experiment!r}, subject={subject!r} "
            f"in {BMI_VELOCITY_SCALERS_PATH} -- available: {available}. Re-run "
            f"compute_velocity_scalers.py if this subject's data is new.")
    entry = scalers[experiment][subject]
    return entry["v_lo"], entry["v_hi"], entry["margin"]


def experiment_and_subject_from_path(path: str) -> Tuple[str, str]:
    """Derives (experiment, subject) from a dataset path like
    '.../bmi/indy/mua_8_group/indy_20160407_02' -> ('bmi', 'indy'),
    matching this project's established {experiment}/{subject}/...
    directory convention (same convention single_subject_pipeline.py's
    own experiment_and_subject_from_raw_stem() derives from a raw_stem
    string instead of a path).

    Deliberately matches against the LOADED SCALERS' OWN top-level keys
    (rather than a separate, hardcoded {"bmi", "hkm"} list that would
    need to be kept in sync by hand) -- finds the first path component
    that is itself a known experiment, and takes the NEXT component as
    the subject. Robust to any depth before that point (however deep
    BMI_DATA_ROOT/BMI_PROJECT_ROOT happen to sit) and any depth after it
    (a group-size directory between subject and session, or not)."""
    scalers = _load_velocity_scalers()
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if part in scalers and i + 1 < len(parts) and parts[i + 1] in scalers[part]:
            return part, parts[i + 1]
    raise ValueError(
        f"Could not derive (experiment, subject) from path={path!r} -- expected an "
        f"'.../{{experiment}}/{{subject}}/...' segment matching one of "
        f"{list(scalers.keys())} followed by one of its known subjects. Pass "
        f"experiment/subject explicitly to CustomDataset if this path doesn't follow "
        f"that convention.")


class CustomDataset(Dataset):
    """
    Dataset for per-trial BMI neural recordings.
    Loads .pt files from a flat directory (train/ or test/) where files are
    named 0.pt, 1.pt, 2.pt, ... with no gaps — as produced by save_bmi_trials.py.
    Args:
        path:  Base directory containing 'train/' and 'test/' subfolders.
        train: Whether to load from the 'train/' (True) or 'test/' (False) subfolder.
        experiment, subject: Which velocity scalers to use (see
            get_velocity_scalers()). Both default to None, in which case
            they're auto-derived from `path` itself (see
            experiment_and_subject_from_path()) -- pass them explicitly
            only if `path` doesn't follow the standard
            '.../{experiment}/{subject}/...' convention.
    """
    def __init__(self, path: str = '../datasets/bmi/', train: bool = True,
                 experiment: Optional[str] = None, subject: Optional[str] = None):
        super().__init__()
        self.split_dir = os.path.join(path, 'train' if train else 'test')
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        # Collect and sort .pkl files numerically (0.pkl < 1.pkl < 2.pkl ...)
        files = [f for f in os.listdir(self.split_dir) if f.endswith('.pkl')]
        files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
        self.files = files

        if experiment is None or subject is None:
            derived_experiment, derived_subject = experiment_and_subject_from_path(path)
            experiment = experiment if experiment is not None else derived_experiment
            subject = subject if subject is not None else derived_subject
        self.experiment = experiment
        self.subject = subject
        self.v_lo, self.v_hi, self.v_margin = get_velocity_scalers(experiment, subject)

        print(f"Loaded {'train' if train else 'test'} split: {len(self.files)} trials from "
              f"{self.split_dir} (experiment={experiment}, subject={subject}, "
              f"v_lo={self.v_lo:.2f}, v_hi={self.v_hi:.2f}, margin={self.v_margin})")

    def __len__(self) -> int:
        return len(self.files)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        spikes   : FloatTensor (time_bins, 2, 96)  — binary spike raster
        position : FloatTensor (time_bins, 2)       — hand [x, y]
        markers  : FloatTensor (n_markers,)         — event times in ms relative to go cue
        """
        data = pkl.load(open(os.path.join(self.split_dir, self.files[idx]), 'rb'))
        
        spikes = torch.from_numpy(data['input_spikes'].T > 0).float()
        # Scaling MUST be the exact forward transform whose inverse is
        # train_bmi.py's unscale_velocity() -- train_bmi.py must be
        # called with THIS SAME dataset's own v_lo/v_hi/margin (e.g. via
        # --velocity-lo/--velocity-hi/--velocity-margin, sourced from
        # this same get_velocity_scalers() call) for the two to actually
        # match; nothing here enforces that automatically across the
        # process boundary between this dataloader and train_bmi.py's
        # own CLI args. Uses THIS INSTANCE's own self.v_lo/self.v_hi/
        # self.v_margin (looked up per experiment/subject in __init__),
        # not the legacy, Indy-only module-level V_LO/V_HI/V_MARGIN --
        # see those constants' own comment for why they're kept around
        # but no longer used here.
        velocity = torch.from_numpy(
            self.v_margin + (1 - 2 * self.v_margin) * (data['velocity'] - self.v_lo) / (self.v_hi - self.v_lo)
        ).float()
        label = torch.tensor([float('nan')])
        
        return label, spikes, velocity
def collate_fn(
    batch: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of (spikes, position) tuples into batch tensors."""
    labels, spikes, position = zip(*batch)
    return torch.stack(labels), torch.stack(spikes), torch.stack(position)
def create_dataloaders(
    data_path: str = '../datasets/bmi/',
    batch_size: int = 20,
    num_workers: int = 4,
    shuffle_train: bool = True,
    drop_last: bool = True,
    small: bool = False,
    experiment: Optional[str] = None,
    subject: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for the BMI dataset.

    NOTE: shuffle_train's default is True again -- reverted back from a
    prior revision that changed it to False. That change existed
    specifically because training used continuous, cross-trial membrane-
    potential state (see SNN_Speck.forward()'s reset_state parameter),
    where shuffling would silently break continuity by feeding genuinely
    out-of-order trials with state improperly carried between them.
    train_bmi.py has since reverted to resetting state on EVERY trial
    (back to the original windowed paradigm, each trial independent and
    self-contained), so that constraint no longer applies -- shuffling is
    safe, and generally preferable (standard practice for i.i.d.-style
    training, avoiding ordering-related gradient artifacts). Pass
    shuffle_train=False explicitly if a script specifically needs strict
    chronological order for some OTHER reason (e.g.
    make_large_snn_dataset.py, which reads through a session once to
    concatenate consecutive trials, and check_velocity_continuity.py,
    which specifically checks chronological ordering).

    experiment, subject: passed straight through to both CustomDataset
    calls below (both train and test splits of the SAME session must
    obviously use the SAME scalers) -- default None auto-derives from
    data_path, same as CustomDataset itself; see
    experiment_and_subject_from_path().

    Args:
        data_path:     Base directory containing 'train/' and 'test/' subfolders.
        batch_size:    Number of trials per batch.
        num_workers:   Worker processes for parallel loading.
        shuffle_train: Whether to shuffle the training split each epoch.
        drop_last:     Whether to drop the final incomplete batch.
    Returns:
        (train_loader, test_loader)
    """
    train_dataset = CustomDataset(path=data_path, train=True, experiment=experiment, subject=subject)
    test_dataset  = CustomDataset(path=data_path, train=False, experiment=experiment, subject=subject)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader
