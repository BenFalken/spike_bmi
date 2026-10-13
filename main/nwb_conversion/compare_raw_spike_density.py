"""
Compare the raw spike data of HKM NWB sessions (Jenkins/Nitschke) with the
BMI .mat sessions (Indy): is HKM's denser SNN input real, or a time-base
error (a wrong sampling rate squeezing the spikes closer together)?

The pipeline never assumes a sampling rate for spikes: NWB spike times and
hand timestamps are read in seconds (convert_nwb_trials_to_raw_h5.py), the
.mat spike times likewise (process_data.py), and both are binned at 4 ms. So
spikes can only be too dense if the times in the file itself are off. Each
of these would show it:

  clocks       the spike times should span the hand timestamps; spikes on a
               clock 1000/676 = 1.48x too fast would end about a third of
               the way before (or after) the hand data
  hand rate    the median hand sampling interval and the rate it implies
  ISI floor    threshold crossings have a hardware dead time (about 1 ms or
               more on a Cerebus), so the shortest inter-spike intervals sit
               on a floor; a compressed time base moves that floor down by
               the same factor (1 ms -> 0.68 ms), so compare it with Indy's
  lag          neural activity leads hand speed by roughly 50-150 ms; the
               lag of the peak cross-correlation of population rate and
               speed is computed over the first and the last third of the
               session: on a wrong clock it drifts by minutes, not ms

and the densities themselves: firing rate per channel (whole session and,
for NWB, inside trials, where HKM's datasets come from), input events per 4
ms step, and the share of 4 ms bins with more than one spike.

Only h5py and numpy are needed (an NWB file is HDF5), so this runs wherever
the raw files are, e.g. on the cluster:

    python compare_raw_spike_density.py \\
        --mat  $DATA_ROOT/raw/bmi/indy/indy_20160407_02.mat \\
        --nwb  $DATA_ROOT/raw/hkm/jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb \\
        --snn-dataset $DATA_ROOT/snn_datasets/bmi/indy/mua_8_group/indy_20160407_02 \\
                      $DATA_ROOT/snn_datasets/hkm/jenkins/mua/sub-Jenkins_ses-20090912_behavior+ecephys \\
        --output spike_density_comparison.json

Several --mat and --nwb files can be given; each is one column of the table.
"""

import argparse
import glob
import json
import os
import pickle

import h5py
import numpy as np

BIN_S = 0.004              # the SNN timestep of every dataset
MAX_LAG_S = 0.5
SMOOTH_S = 0.02            # Gaussian smoothing of population rate and speed for the lag
ISI_LIMITS_MS = (0.5, 1.0, 1.6)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_mat(path):
    """Indy-style .mat (v7.3): MUA per channel (every unit of a channel
    merged, as process_data.py does), cursor position and its times."""
    with h5py.File(path, 'r') as f:
        t = f['t'][()].squeeze()
        xy = f['cursor_pos'][()].T[:, :2]
        refs = f['spikes'][()].T                       # (channels, units)
        channels, units = [], []
        for c in range(refs.shape[0]):
            spikes = [np.sort(f[refs[c, u]][()].ravel()) for u in range(refs.shape[1])
                      if f[refs[c, u]].ndim == 2]
            channels.append(np.sort(np.concatenate(spikes)) if spikes else np.empty(0))
            units += spikes
    # Merged channels have no dead time (two units can fire together), so
    # the ISI floor is measured on the units as recorded.
    return {'kind': 'mat', 'units': channels, 'isi_units': units, 'hand_t': t, 'hand_xy': xy,
            'hand_time_source': 't', 'intervals': None}


def _find_hand(f):
    preferred = 'processing/behavior/Position/Hand'
    if preferred in f:
        return f[preferred]
    found = []
    f.visit(lambda name: found.append(name) if name.endswith('/Hand') and isinstance(f[name], h5py.Group)
            else None)
    if not found:
        raise KeyError("no Hand position series in the NWB file")
    return f[found[0]]


def load_nwb(path):
    """NWB: every unit's spike times (one per electrode in these sessions),
    hand position and its times, and the trials table."""
    with h5py.File(path, 'r') as f:
        times = f['units/spike_times'][()]
        ends = f['units/spike_times_index'][()]
        units = [np.sort(u) for u in np.split(times, ends[:-1])]
        hand = _find_hand(f)
        xy = hand['data'][()][:, :2]
        if 'timestamps' in hand:
            t, source = hand['timestamps'][()], 'timestamps'
        else:                                          # starting_time + rate
            start = hand['starting_time']
            t = start[()] + np.arange(len(xy)) / start.attrs['rate']
            source = f"starting_time + rate ({start.attrs['rate']:g} Hz)"
        intervals = None
        if 'intervals/trials/start_time' in f:
            intervals = np.stack([f['intervals/trials/start_time'][()],
                                  f['intervals/trials/stop_time'][()]], axis=1)
    return {'kind': 'nwb', 'units': units, 'hand_t': t, 'hand_xy': xy, 'hand_time_source': source,
            'intervals': intervals}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def clock_metrics(data):
    t = data['hand_t']
    dt = np.diff(t)
    spikes = [u for u in data['units'] if len(u)]
    first = min(u[0] for u in spikes)
    last = max(u[-1] for u in spikes)
    return {'hand_samples': int(len(t)),
            'hand_time_source': data['hand_time_source'],
            'hand_dt_median_ms': 1000 * float(np.median(dt)),
            'hand_rate_hz': float(1 / np.median(dt)),
            'hand_dt_p1_p99_ms': [1000 * _pct(dt, 1), 1000 * _pct(dt, 99)],
            'hand_gaps_over_50ms': int(np.sum(dt > 0.05)),
            'hand_span_s': [float(t[0]), float(t[-1])],
            'spike_span_s': [float(first), float(last)],
            'spike_span_over_hand_span': float((last - first) / (t[-1] - t[0])),
            'trials_span_s': (None if data['intervals'] is None else
                              [float(data['intervals'][:, 0].min()), float(data['intervals'][:, 1].max())])}


def _in_intervals(spikes, intervals):
    keep = np.zeros(len(spikes), dtype=bool)
    for lo, hi in intervals:
        a, b = np.searchsorted(spikes, [lo, hi])
        keep[a:b] = True
    return spikes[keep]


def density_metrics(units, intervals):
    """Rates (Hz per channel), events per 4 ms step, and the share of 4 ms
    bins with more than one spike, over the given [start, stop] intervals."""
    duration = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    inside = [_in_intervals(u, intervals) for u in units]
    rates = np.array([len(u) / duration for u in inside])
    multi, nonzero = 0, 0
    for u in inside:                                   # 4 ms bins of every interval
        for lo, hi in intervals:
            a, b = np.searchsorted(u, [lo, hi])
            if b <= a:
                continue
            counts = np.bincount(((u[a:b] - lo) // BIN_S).astype(int))
            multi += int(np.sum(counts > 1))
            nonzero += int(np.sum(counts > 0))
    return {'duration_s': duration,
            'channels': int(len(units)),
            'rate_hz_mean': float(rates.mean()),
            'rate_hz_median': float(np.median(rates)),
            'rate_hz_p10_p90': [_pct(rates, 10), _pct(rates, 90)],
            'silent_channels': int(np.sum(rates == 0)),
            'events_per_4ms_step': float(rates.sum() * BIN_S),
            'nonzero_bins_with_multiple_spikes': (multi / nonzero) if nonzero else None}


def isi_metrics(units):
    """Inter-spike intervals within each channel: the floor the recording
    hardware leaves, and the share of very short intervals."""
    isis = [np.diff(u) * 1000 for u in units if len(u) > 100]
    pooled = np.concatenate(isis) if isis else np.empty(0)
    per_channel_floor = [float(np.percentile(i, 0.1)) for i in isis]
    hist_edges = np.arange(0, 3.05, 0.1)
    hist = np.histogram(pooled, hist_edges)[0]
    return {'channels_used': len(isis),
            'isi_floor_ms_median_over_channels': float(np.median(per_channel_floor)) if isis else None,
            'isi_ms_p0.1_p1_pooled': [_pct(pooled, 0.1), _pct(pooled, 1)],
            **{f'share_isi_under_{lim:g}ms': float(np.mean(pooled < lim)) if len(pooled) else None
               for lim in ISI_LIMITS_MS},
            'isi_hist_0_to_3ms_0.1ms_bins': hist.tolist(),
            'isi_shortest_occupied_bin_ms': (float(hist_edges[np.argmax(hist > 0.001 * hist.sum())])
                                             if hist.sum() else None)}


def _smooth(x, sigma_bins):
    half = int(4 * sigma_bins)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_bins) ** 2)
    return np.convolve(x, k / k.sum(), mode='same')


def _zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def lag_metrics(data):
    """Lag (s) at which population rate best matches hand speed, over the
    first and last third of the session; positive = neural activity leads.
    Grid points more than 20 ms from a hand sample are left out."""
    t, xy = data['hand_t'], data['hand_xy']
    grid = np.arange(t[0], t[-1], BIN_S)
    nearest = np.abs(t[np.clip(np.searchsorted(t, grid), 0, len(t) - 1)] - grid)
    valid = nearest < 0.02
    pos = np.stack([np.interp(grid, t, xy[:, k]) for k in range(2)], axis=1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / BIN_S]
    pop = np.zeros(len(grid))
    for u in data['units']:
        idx = ((u[(u >= grid[0]) & (u < grid[-1] + BIN_S)] - grid[0]) // BIN_S).astype(int)
        pop += np.bincount(idx, minlength=len(grid))[:len(grid)]
    sigma = SMOOTH_S / BIN_S
    pop, speed = _smooth(pop, sigma), _smooth(speed, sigma)
    max_lag = int(MAX_LAG_S / BIN_S)

    def best_lag(lo, hi):
        p, s, v = pop[lo:hi], speed[lo:hi], valid[lo:hi]
        best = (None, -np.inf)
        for lag in range(-max_lag, max_lag + 1):   # pop[i] against speed[i + lag]
            a, b = max(0, -lag), min(len(p), len(p) - lag)
            m = v[a:b] & v[a + lag:b + lag]
            if m.sum() < 1000:
                continue
            cc = float(np.mean(_zscore(p[a:b][m]) * _zscore(s[a + lag:b + lag][m])))
            if cc > best[1]:
                best = (lag * BIN_S, cc)
        return {'lag_s': best[0], 'cc': best[1] if best[0] is not None else None}

    n = len(grid)
    return {'first_third': best_lag(0, n // 3), 'last_third': best_lag(2 * n // 3, n)}


def snn_dataset_metrics(path):
    files = sorted(glob.glob(os.path.join(path, 'train', '*.pkl')) + glob.glob(os.path.join(path, 'test', '*.pkl')))
    if not files:
        files = sorted(glob.glob(os.path.join(path, '*.pkl')))
    events, steps, multi, nonzero, channels = 0.0, 0, 0, 0, None
    for name in files:
        with open(name, 'rb') as f:
            x = np.asarray(pickle.load(f)['input_spikes'])
        x = x.reshape(-1, *x.shape[-2:])               # (windows or 1, channels, T)
        channels = x.shape[1]
        events += float(x.sum())
        steps += x.shape[0] * x.shape[2]
        multi += int(np.sum(x > 1))
        nonzero += int(np.sum(x > 0))
    return {'files': len(files), 'channels': channels,
            'events_per_step': events / steps if steps else None,
            'rate_hz_mean': events / steps / channels / BIN_S if steps else None,
            'nonzero_bins_with_multiple_spikes': multi / nonzero if nonzero else None}


def session_metrics(path):
    data = load_nwb(path) if path.endswith('.nwb') else load_mat(path)
    t = data['hand_t']
    out = {'file': path, 'kind': data['kind'], 'clocks': clock_metrics(data),
           'density_whole_session': density_metrics(data['units'], np.array([[t[0], t[-1]]])),
           'isi': isi_metrics(data.get('isi_units', data['units'])), 'lag': lag_metrics(data)}
    if data['intervals'] is not None:
        out['density_in_trials'] = density_metrics(data['units'], data['intervals'])
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    if isinstance(v, list):
        return '/'.join(_fmt(x) for x in v)
    return str(v)


ROWS = [
    ('hand rate (Hz)', lambda s: s['clocks']['hand_rate_hz']),
    ('hand dt p1/p99 (ms)', lambda s: s['clocks']['hand_dt_p1_p99_ms']),
    ('hand time from', lambda s: s['clocks']['hand_time_source']),
    ('hand span (s)', lambda s: s['clocks']['hand_span_s']),
    ('spike span (s)', lambda s: s['clocks']['spike_span_s']),
    ('spike span / hand span', lambda s: s['clocks']['spike_span_over_hand_span']),
    ('trials span (s)', lambda s: s['clocks']['trials_span_s']),
    ('channels', lambda s: s['density_whole_session']['channels']),
    ('rate Hz, session (mean)', lambda s: s['density_whole_session']['rate_hz_mean']),
    ('rate Hz, in trials (mean)', lambda s: s.get('density_in_trials', {}).get('rate_hz_mean')),
    ('events/4 ms, session', lambda s: s['density_whole_session']['events_per_4ms_step']),
    ('events/4 ms, in trials', lambda s: s.get('density_in_trials', {}).get('events_per_4ms_step')),
    ('bins >1 spike, session', lambda s: s['density_whole_session']['nonzero_bins_with_multiple_spikes']),
    ('ISI floor (ms, median ch)', lambda s: s['isi']['isi_floor_ms_median_over_channels']),
    ('shortest ISI bin (ms)', lambda s: s['isi']['isi_shortest_occupied_bin_ms']),
    ('ISI < 1 ms (share)', lambda s: s['isi']['share_isi_under_1ms']),
    ('lag first third (s, cc)', lambda s: [s['lag']['first_third']['lag_s'], s['lag']['first_third']['cc']]),
    ('lag last third (s, cc)', lambda s: [s['lag']['last_third']['lag_s'], s['lag']['last_third']['cc']]),
]


def print_table(sessions, snn):
    names = [os.path.splitext(os.path.basename(s['file']))[0][:28] for s in sessions]
    width = max([28] + [len(n) for n in names]) + 2
    print('\n' + ' ' * 28 + ''.join(n.rjust(width) for n in names))
    for label, get in ROWS:
        print(label.ljust(28) + ''.join(_fmt(get(s)).rjust(width) for s in sessions))
    for path, m in snn.items():
        print(f"\nSNN dataset {path}: {m['files']} files, {m['channels']} channels, "
              f"{_fmt(m['events_per_step'])} events/step ({_fmt(m['rate_hz_mean'])} Hz per channel), "
              f"nonzero bins with >1 spike {_fmt(m['nonzero_bins_with_multiple_spikes'])}")


def main(args):
    sessions = []
    for path in args.mat + args.nwb:
        print(f"Reading {path}", flush=True)
        sessions.append(session_metrics(path))
    snn = {p: snn_dataset_metrics(p) for p in args.snn_dataset}
    print_table(sessions, snn)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'sessions': sessions, 'snn_datasets': snn}, f, indent=2)
        print(f"\nSaved {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mat', nargs='*', default=[], help='BMI (Indy) raw .mat sessions')
    parser.add_argument('--nwb', nargs='*', default=[], help='HKM raw .nwb sessions')
    parser.add_argument('--snn-dataset', nargs='*', default=[],
                        help='SNN dataset session directories (train/ and test/ .pkl) to compare')
    parser.add_argument('--output', default=None, help='JSON with every metric')
    main(parser.parse_args())
