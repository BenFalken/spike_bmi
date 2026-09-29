"""
List of preprocessing functions
"""

# import packages
import numpy as np
from scipy import signal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import _num_samples

def filter(x, fc, fs, order, btype="lowpass", zero_phase=True):
    """
    Filter data with a Butterworth filter.

    Parameters
    ----------
    x : ndarray
        The data to be filtered.
    fc : float or list
        The critical frequency.
    fs : float
        The sampling frequency.
    order : int
        The order of the filter.
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default 'lowpass'
        The type of filter.
    zero_phase : bool, default True.
        Zero phase (forward backward) filter.

    Returns
    ----------
    y : ndarray
        The filtered data.
    """
    fnyq = fs/2 # Nyquist frequency
    if btype in ['bandpass', 'bandstop']:
        assert len(fc)==2, f"for {btype}, you should provide a sequence of two frequencies (low and high)"
        Wn = [f/fnyq for f in fc]
    else:
        Wn = fc/fnyq
    
    b, a = signal.butter(order, Wn, btype=btype)
    if zero_phase:
        y = signal.filtfilt(b, a, x)
    else:
        y = signal.lfilter(b, a, x)
    return y
    
def downsample(x, k):
    """
    Downsample data.

    Parameters
    ----------
    x : ndarray
        The data to be downsampled.
    k : int
        The downsample factor.

    Returns
    ----------
    y : ndarray
        The downsampled data.

    """
    N = len(x)
    idx = np.arange(0, N, k)
    y = x[idx]
    return y


class TimeSeriesSplitCustom(TimeSeriesSplit):
    """
    Chronological cross-validation splitter for time-series data, with an
    optional purge GAP between each fold's training and test indices.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits.
    max_train_size : int, default=None
        Maximum size for a single training set.
    test_size : int, default=1
        Size of each test fold, in samples.
    min_train_size : int, default=1
        Minimum number of training samples required in the first
        (smallest) fold. Checked after folds are generated; raises
        ValueError if violated.
    gap : int, default=0
        Number of samples excluded between each fold's training and test
        indices.

    Returns
    ----------
    Indices of training and testing data, via .split(X, y).
    """
    def __init__(self, n_splits=5, max_train_size=None, test_size=1,
                 min_train_size=1, gap=0):
        super().__init__(n_splits=n_splits, max_train_size=max_train_size,
                          test_size=test_size, gap=gap)
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        for fold_i, (train_idx, test_idx) in enumerate(super().split(X, y, groups)):
            if fold_i == 0 and len(train_idx) < self.min_train_size:
                raise ValueError(
                    f"First fold's training size ({len(train_idx)} samples) is "
                    f"smaller than min_train_size ({self.min_train_size}). Reduce "
                    f"min_train_size, reduce test_size/gap, or provide more data.")
            yield train_idx, test_idx


def aligned_train_boundary(total_raw_samples, test_frac, base_nperseg=65):
    """
    Number of raw samples in the training split, rounded down to a
    multiple of base_nperseg.

    Every dataset built from a session (dense ANN windows, non-overlapping
    SNN windows, grouped SNN trials) computes its train/test boundary with
    this function from the same raw session length, so all of them hold out
    the same stretch of recording and no SNN window straddles the boundary.

    Parameters
    ----------
    total_raw_samples : int
        Length of the unwindowed session (len(task_time)).
    test_frac : float
        Fraction of the session held out as test (taken from the end).
    base_nperseg : int, default=65
        SNN window length in raw samples (256 ms at 4 ms/sample).

    Returns
    ----------
    n_train_raw : int
    """
    n_test_raw = round(test_frac * total_raw_samples)
    n_train_raw = total_raw_samples - n_test_raw
    return (n_train_raw // base_nperseg) * base_nperseg


def chronological_holdout_split(X, y, test_frac, gap=0, base_nperseg=65, n_train_override=None):
    """
    Single chronological train/test split of a dense-windowed ANN dataset,
    with the boundary aligned to the SNN windows (see aligned_train_boundary).

    Parameters
    ----------
    X, y : ndarray
        Full, chronologically-ordered dataset. Rows must come from step=1
        (4 ms) windowing, so that row index equals raw sample index and
        len(X) == total_raw_samples - base_nperseg.
    test_frac : float
        Fraction of the session held out as test.
    gap : int, default=0
        Rows dropped from the end of the training set (purge gap against
        leakage between overlapping windows).
    base_nperseg : int, default=65
        SNN window length in raw samples.
    n_train_override : int, optional
        Use this row count as the boundary instead. Required for
        trial-structured (NWB) datasets, whose rows are concatenated trials
        rather than one continuous session.

    Returns
    ----------
    X_train, y_train, X_test, y_test : ndarray
    """
    n_samples = _num_samples(X)
    if n_train_override is not None:
        n_train = n_train_override
    else:
        n_train = aligned_train_boundary(n_samples + base_nperseg, test_frac, base_nperseg)

    n_train_end = n_train - gap
    if n_train_end <= 0:
        raise ValueError(
            f"test_frac={test_frac} and gap={gap} leave no training data "
            f"out of {n_samples} samples.")
    X_train, y_train = X[:n_train_end], y[:n_train_end]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test


def transform_data(X, y, timesteps):
    """
    Stack `timesteps` consecutive rows of X into one example, paired with
    the target at the example's last row.

    Parameters
    ----------
    X : ndarray
        The input data
    y : ndarray
        The output (target) data
    timesteps: int
        The number of input steps to predict next step

    Returns
    ----------
    X_seq : ndarray
        The transformed input sequence data
    y_seq : ndarray
        The transformed ouput (target) sequence data
    """
    X_seq = []
    y_seq = []
    # check length X_in equals to y_in
    assert len(X) == len(y), "Both input data length must be equal"
    for i in range(len(X)):
        end_idx = i + timesteps
        if end_idx > len(X)-1:
            break # break if index exceeds the data length
        # get input and output sequence
        X_seq.append(X[i:end_idx,:])
        y_seq.append(y[end_idx-1,:])
    return np.asarray(X_seq), np.asarray(y_seq)
