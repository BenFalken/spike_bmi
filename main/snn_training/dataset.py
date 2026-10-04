"""
Dataset and DataLoaders for the SNN's .pkl trials.

A dataset directory has train/ and test/ subfolders of trials named 0.pkl,
1.pkl, ... (or {session}_{i}.pkl in a pooled directory), each a dict with
    input_spikes  (n_units, T)  spike counts per 4 ms bin
    velocity      (T, 2)        hand velocity in physical units

Velocity is scaled to [margin, 1 - margin] using per-subject bounds from
velocity_scalers.json (written by preprocessing_training/
compute_velocity_scalers.py; override the path with
BMI_VELOCITY_SCALERS_PATH). The experiment and subject are read from the
dataset path, which must contain .../{experiment}/{subject}/... .
"""

import json
import os
import pickle as pkl
import re
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

BMI_VELOCITY_SCALERS_PATH = os.environ.get(
    "BMI_VELOCITY_SCALERS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_scalers.json"),
)

_velocity_scalers_cache = None


def _load_velocity_scalers():
    """{experiment: {subject: {v_lo, v_hi, margin, ...}}}, read once per process."""
    global _velocity_scalers_cache
    if _velocity_scalers_cache is None:
        if not os.path.isfile(BMI_VELOCITY_SCALERS_PATH):
            raise FileNotFoundError(
                f"{BMI_VELOCITY_SCALERS_PATH} does not exist; run compute_velocity_scalers.py "
                f"or set BMI_VELOCITY_SCALERS_PATH.")
        with open(BMI_VELOCITY_SCALERS_PATH, "r") as f:
            _velocity_scalers_cache = json.load(f)
    return _velocity_scalers_cache


def get_velocity_scalers(experiment: str, subject: str) -> Tuple[float, float, float]:
    """(v_lo, v_hi, margin) for one experiment/subject."""
    scalers = _load_velocity_scalers()
    if experiment not in scalers or subject not in scalers[experiment]:
        available = {exp: list(subs) for exp, subs in scalers.items()}
        raise KeyError(f"No velocity scalers for {experiment}/{subject} in "
                       f"{BMI_VELOCITY_SCALERS_PATH}; available: {available}")
    entry = scalers[experiment][subject]
    return entry["v_lo"], entry["v_hi"], entry["margin"]


def experiment_and_subject_from_path(path: str) -> Tuple[str, str]:
    """'.../bmi/indy/mua_8_group/indy_20160407_02' -> ('bmi', 'indy').

    Finds the first path component that is an experiment in
    velocity_scalers.json and is followed by one of its subjects."""
    scalers = _load_velocity_scalers()
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts[:-1]):
        if part in scalers and parts[i + 1] in scalers[part]:
            return part, parts[i + 1]
    raise ValueError(f"No '.../{{experiment}}/{{subject}}/...' segment in {path!r} matching "
                     f"velocity_scalers.json (experiments: {list(scalers)})")


class CustomDataset(Dataset):
    """One split (train/ or test/) of a dataset directory.

    Items are (label, spikes (T, n_units), scaled velocity (T, 2)); label is
    an unused NaN placeholder kept for the training loop's signature.
    experiment/subject default to the values derived from `path`."""

    def __init__(self, path: str, train: bool = True,
                 experiment: Optional[str] = None, subject: Optional[str] = None):
        super().__init__()
        self.split_dir = os.path.join(path, 'train' if train else 'test')
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        files = [f for f in os.listdir(self.split_dir) if f.endswith('.pkl')]
        self.files = sorted(files, key=_natural_key)

        if experiment is None or subject is None:
            derived_experiment, derived_subject = experiment_and_subject_from_path(path)
            experiment = experiment or derived_experiment
            subject = subject or derived_subject
        self.experiment, self.subject = experiment, subject
        self.v_lo, self.v_hi, self.v_margin = get_velocity_scalers(experiment, subject)
        print(f"Loaded {'train' if train else 'test'} split: {len(self.files)} trials from "
              f"{self.split_dir} ({experiment}/{subject}, v_lo={self.v_lo:.2f}, "
              f"v_hi={self.v_hi:.2f}, margin={self.v_margin})")

    def __len__(self) -> int:
        return len(self.files)

    def trial_length(self, idx: int) -> int:
        """Timesteps in trial idx, without scaling its velocity."""
        with open(os.path.join(self.split_dir, self.files[idx]), 'rb') as f:
            return pkl.load(f)['input_spikes'].shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(os.path.join(self.split_dir, self.files[idx]), 'rb') as f:
            data = pkl.load(f)
        spikes = torch.from_numpy(data['input_spikes'].T).float()
        # Inverted by train_snn.unscale_velocity().
        velocity = torch.from_numpy(
            self.v_margin + (1 - 2 * self.v_margin) * (data['velocity'] - self.v_lo) / (self.v_hi - self.v_lo)
        ).float()
        return torch.tensor([float('nan')]), spikes, velocity


def _natural_key(name: str):
    """'s_10.pkl' sorts after 's_9.pkl'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', name)]


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels, spikes, velocity = zip(*batch)
    lengths = {s.shape[0] for s in spikes}
    if len(lengths) > 1:
        raise ValueError(f"Cannot batch trials of different lengths ({sorted(lengths)[:5]}...). "
                         f"Whole-trial datasets (hkm) need --batch-size 1.")
    return torch.stack(labels), torch.stack(spikes), torch.stack(velocity)


def create_dataloaders(
    data_path: str,
    batch_size: int = 20,
    num_workers: int = 4,
    shuffle_train: bool = True,
    drop_last: bool = True,
    experiment: Optional[str] = None,
    subject: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """(train_loader, test_loader) for one dataset directory; both splits use
    the same velocity scalers."""
    train_dataset = CustomDataset(data_path, train=True, experiment=experiment, subject=subject)
    test_dataset = CustomDataset(data_path, train=False, experiment=experiment, subject=subject)
    common = dict(batch_size=batch_size, num_workers=num_workers, drop_last=drop_last,
                  pin_memory=True, collate_fn=collate_fn)
    return (DataLoader(train_dataset, shuffle=shuffle_train, **common),
            DataLoader(test_dataset, shuffle=False, **common))
