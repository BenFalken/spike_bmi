#!/usr/bin/env python3
"""
diagnose_hkm_sessions.py -- what actually lies inside each HKM session?

For every session directory under --data-root (each with train/ and test/ of
whole-trial .pkl files: input_spikes C x T, velocity T x 2 in physical units) this
produces, per session:

  1. TRIAL CENSUS     counts, length distribution, empty / very short trials,
                      dead channels, firing-rate statistics, NaN/inf checks.
  2. VELOCITY CENSUS  percentiles, outliers, between-trial vs within-trial variance,
                      smoothness, train->test mean shift, and the loss floors that
                      trivial predictors achieve (zero, train-mean, clipped-to-bounds)
                      using the SAME metric as train_hkm.py ("Test Loss" = mean over
                      trials of sqrt(MSE) in physical units).
  3. FIRING <-> HAND  summed-rate vs speed/vx/vy correlation, lagged cross-correlation,
                      per-channel correlation distribution, and a causal ridge-regression
                      ceiling (what a plain linear decoder gets on test).
  4. ALIGNED AVERAGES trial-averaged summed firing vs trial-averaged speed / vx / vy,
                      aligned at trial start, trial end, movement onset and peak speed.

Outputs (in --out-dir):
  <session>_aggregate.png    aligned trial averages
  <session>_census.png       length / velocity histograms, lagged xcorr, channel corr
  <session>_examples.png     a few raw trials (summed rate vs velocity)
  <session>_diagnostics.json per-session results + parameters (completion marker)
  summary_all_sessions.png   cross-session comparison
  diagnostics.json           every number, per session
  diagnostics.txt            human-readable report with warnings

RESUMING: sessions whose <session>_diagnostics.json (and figures) already exist in
--out-dir, computed with the same analysis parameters, are loaded instead of
recomputed. Use --force to recompute everything, or delete a session's
<session>_diagnostics.json to redo just that session.

Usage:
  python diagnose_hkm_sessions.py \
      --data-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/hkm/nitschke/mua \
      --out-dir hkm_diagnostics/nitschke
  # optional: --v-lo/--v-hi to compare against the scaling bounds the trainer used
  #           (printed in the train_hkm.py log as v_lo / v_hi)

Only needs numpy + matplotlib (torch is used only if a pkl stores tensors).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- loading
def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_split(split_dir):
    """Return list of dicts {name, spikes (C,T) float32, vel (T,2) float64}."""
    trials = []
    for path in sorted(glob.glob(os.path.join(split_dir, "*.pkl"))):
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            s = _to_np(d["input_spikes"]).astype(np.float32)
            v = _to_np(d["velocity"]).astype(np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not read {path}: {e}")
            continue
        if v.ndim == 2 and v.shape[0] == 2 and v.shape[1] != 2:
            v = v.T
        T = v.shape[0]
        if T < 2:
            print(f"  [warn] {path}: only {T} sample(s), skipped")
            continue
        if s.ndim != 2:
            print(f"  [warn] {path}: spikes ndim={s.ndim}, skipped")
            continue
        if s.shape[1] != T and s.shape[0] == T:
            s = s.T
        if s.shape[1] != T:
            print(f"  [warn] {path}: spikes T={s.shape[1]} != velocity T={T}; cropping to min")
            m = min(s.shape[1], T)
            s, v = s[:, :m], v[:m]
        trials.append({"name": os.path.basename(path), "spikes": s, "vel": v})
    return trials


# --------------------------------------------------------------------------- helpers
def box_smooth(x, w):
    if w <= 1 or len(x) < 2:
        return x.astype(np.float64)
    w = min(w, len(x))  # np.convolve(mode="same") returns max(len(x), len(k)) -> wrong length for T < w
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def pct(a, qs=(0, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 100)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def trial_loss(pred, gt):
    """train_hkm 'Test Loss' per trial: sqrt(mean squared error over T and both axes)."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def summed_rate(trial, smooth):
    return box_smooth(trial["spikes"].sum(axis=0), smooth)


def speed(v):
    return np.sqrt((v ** 2).sum(axis=1))


def nanmean_sem(stack, min_count):
    """stack: (n_trials, L) with NaN padding."""
    cnt = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(all="ignore"):
        m = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
    sem = sd / np.sqrt(np.maximum(cnt, 1))
    bad = cnt < min_count
    m[bad] = np.nan
    sem[bad] = np.nan
    return m, sem, cnt


# --------------------------------------------------------------------------- resume / IO helpers
# Analysis parameters that change the numbers or figures. A cached session is reused only
# if it was computed with the same values.
PARAM_KEYS = ("smooth", "max_lag", "window", "onset_frac", "min_window", "v_lo", "v_hi", "dt_ms")
FIG_KEYS = ("aggregate", "census", "examples")


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def atomic_json_dump(obj, path):
    """Write JSON via a temp file + rename so a killed job never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def current_params(args):
    return {k: getattr(args, k) for k in PARAM_KEYS}


def session_paths(out_dir, name):
    return {
        "json": os.path.join(out_dir, f"{name}_diagnostics.json"),
        "aggregate": os.path.join(out_dir, f"{name}_aggregate.png"),
        "census": os.path.join(out_dir, f"{name}_census.png"),
        "examples": os.path.join(out_dir, f"{name}_examples.png"),
    }


def load_legacy_combined(out_dir):
    """Results from a combined diagnostics.json written by a previous (completed) run, if any."""
    path = os.path.join(out_dir, "diagnostics.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception as e:  # noqa: BLE001
        print(f"[resume] could not read existing {path} ({e}); ignoring it")
        return {}


def load_completed(name, out_dir, args, legacy):
    """Return (result, reason). result is None if the session must be (re)computed."""
    p = session_paths(out_dir, name)
    figs_ok = all(os.path.isfile(p[k]) for k in FIG_KEYS)
    if os.path.isfile(p["json"]):
        try:
            with open(p["json"]) as f:
                blob = json.load(f)
            res = blob["result"]
        except Exception as e:  # noqa: BLE001
            return None, f"cached json unreadable ({e})"
        stored = blob.get("params")
        if stored is not None:
            stored = {k: stored.get(k) for k in PARAM_KEYS}
            if stored != current_params(args):
                diff = [k for k in PARAM_KEYS if stored[k] != getattr(args, k)]
                return None, f"parameters changed ({', '.join(diff)})"
        if "velocity" in res and not figs_ok:
            return None, "figures missing"
        return res, "cached"
    if name in legacy and "velocity" in legacy[name] and figs_ok:
        # Adopt results from an old combined diagnostics.json and write the per-session marker.
        atomic_json_dump({"params": None, "result": legacy[name]}, p["json"])
        return legacy[name], "adopted from existing diagnostics.json (parameters unverified)"
    return None, "not done"


# --------------------------------------------------------------------------- analyses
def trial_census(trials, min_win):
    if not trials:
        return {"n_trials": 0}
    lens = np.array([t["spikes"].shape[1] for t in trials])
    tot = np.array([t["spikes"].sum() for t in trials])
    C = trials[0]["spikes"].shape[0]
    ch_tot = sum(t["spikes"].sum(axis=1) for t in trials)
    ch_len = lens.sum()
    ch_rate = ch_tot / max(ch_len, 1)
    nonfinite = int(sum((~np.isfinite(t["spikes"])).sum() + (~np.isfinite(t["vel"])).sum() for t in trials))
    spk_per_step = tot / np.maximum(lens, 1)
    return {
        "n_trials": int(len(trials)),
        "n_channels": int(C),
        "total_samples": int(lens.sum()),
        "length_percentiles": pct(lens),
        "n_trials_len_le_%d" % min_win: int((lens <= min_win).sum()),
        "frac_trials_len_le_%d" % min_win: float((lens <= min_win).mean()),
        "n_empty_trials(no spikes)": int((tot == 0).sum()),
        "dead_channels(no spikes in split)": int((ch_tot == 0).sum()),
        "frac_dead_channels": float((ch_tot == 0).mean()),
        "spikes_per_step_per_trial_percentiles": pct(spk_per_step),
        "channel_rate_percentiles(spikes/sample)": pct(ch_rate),
        "spike_value_max": float(max(t["spikes"].max() for t in trials)),
        "n_nonfinite_values": nonfinite,
    }


def velocity_census(train, test, v_lo, v_hi, min_win):
    out = {}
    for tag, trials in (("train", train), ("test", test)):
        if not trials:
            continue
        V = np.concatenate([t["vel"] for t in trials], axis=0)
        sp = speed(V)
        per_trial_mean = np.stack([t["vel"].mean(axis=0) for t in trials])
        lens = np.array([len(t["vel"]) for t in trials])
        tot_var = V.var(axis=0)
        # between-trial share of variance (trial-constant component), weighted by length
        gm = V.mean(axis=0)
        btw = (lens[:, None] * (per_trial_mean - gm) ** 2).sum(axis=0) / lens.sum()
        lag1 = []
        for t in trials:
            for k in range(2):
                lag1.append(pearson(t["vel"][:-1, k], t["vel"][1:, k]) if len(t["vel"]) > 5 else np.nan)
        d = {
            "mean": V.mean(axis=0).tolist(),
            "std": V.std(axis=0).tolist(),
            "vx_percentiles": pct(V[:, 0]),
            "vy_percentiles": pct(V[:, 1]),
            "speed_percentiles": pct(sp),
            "abs_max": float(np.abs(V).max()),
            "between_trial_var_fraction(vx,vy)": (btw / np.maximum(tot_var, 1e-12)).tolist(),
            "lag1_autocorr_median(smoothness)": float(np.nanmedian(lag1)) if len(lag1) else None,
            "n_zero_variance_trials": int(sum(np.all(t["vel"].std(axis=0) < 1e-9) for t in trials)),
            "first_sample_abs_vel_median": float(np.median([np.abs(t["vel"][0]).max() for t in trials])),
        }
        # outliers: how heavy is the tail vs the bulk?
        p99 = np.percentile(np.abs(V), 99)
        d["frac_samples_abs_gt_5x_p99"] = float(np.mean(np.abs(V) > 5 * max(p99, 1e-9)))
        if v_lo is not None and v_hi is not None:
            out_of = (V < v_lo) | (V > v_hi)
            d["frac_samples_outside_[v_lo,v_hi]"] = float(out_of.mean())
            clipped_floor = [trial_loss(np.clip(t["vel"], v_lo, v_hi), t["vel"]) for t in trials]
            d["loss_floor_from_clipping_to_bounds(mean of per-trial sqrtMSE)"] = float(np.mean(clipped_floor))
        out[tag] = d
    # trivial-predictor losses on the eval split with the train-mean
    ev = test if test else train
    ev_tag = "test" if test else "train(no test split)"
    if train and ev:
        mu = np.concatenate([t["vel"] for t in train], axis=0).mean(axis=0)
        zero = np.mean([trial_loss(np.zeros_like(t["vel"]), t["vel"]) for t in ev])
        mean = np.mean([trial_loss(np.tile(mu, (len(t["vel"]), 1)), t["vel"]) for t in ev])
        oracle_trial_mean = np.mean([trial_loss(np.tile(t["vel"].mean(0), (len(t["vel"]), 1)), t["vel"]) for t in ev])
        out["trivial_predictor_loss_on_" + ev_tag] = {
            "predict_zero": float(zero),
            "predict_train_mean": float(mean),
            "predict_per_trial_mean(oracle)": float(oracle_trial_mean),
            "note": "If these are already in the hundreds, the TARGET SCALE is the problem, not the decoder.",
        }
        if train and test:
            tr_m = np.concatenate([t["vel"] for t in train]).mean(0)
            te_m = np.concatenate([t["vel"] for t in test]).mean(0)
            tr_s = np.concatenate([t["vel"] for t in train]).std(0)
            out["train_test_mean_shift_in_train_std_units"] = ((te_m - tr_m) / np.maximum(tr_s, 1e-9)).tolist()
    return out


def lagged_xcorr(trials, smooth, max_lag, key):
    """mean over trials of corr(rate[t], target[t+k]); k>0 => neural LEADS hand."""
    lags = np.arange(-max_lag, max_lag + 1)
    res = np.full((len(trials), len(lags)), np.nan)
    for i, t in enumerate(trials):
        r = summed_rate(t, smooth)
        y = {"speed": speed(t["vel"]), "vx": t["vel"][:, 0], "vy": t["vel"][:, 1]}[key]
        T = len(r)
        for j, k in enumerate(lags):
            if k >= 0:
                a, b = r[: T - k], y[k:]
            else:
                a, b = r[-k:], y[: T + k]
            # Trials shorter than |lag| make these slices wrap (negative end index) and come out
            # mismatched/empty -- that crashed the run on short trials. Require a real overlap.
            if abs(k) < T and len(a) == len(b) and len(a) >= 20:
                res[i, j] = pearson(a, b)
    with np.errstate(all="ignore"):
        mean = np.nanmean(res, axis=0)
    return lags, mean, res


def firing_hand_correlation(trials, smooth, max_lag):
    out = {}
    curves = {}
    for key in ("speed", "vx", "vy"):
        lags, mean, per_trial = lagged_xcorr(trials, smooth, max_lag, key)
        zero_i = int(np.where(lags == 0)[0][0])
        if np.all(np.isnan(mean)):
            out[key] = {"corr_lag0": None}
            curves[key] = (lags, mean)
            continue
        bi = int(np.nanargmax(np.abs(mean)))
        out[key] = {
            "corr_lag0_mean_over_trials": float(mean[zero_i]),
            "best_lag_samples": int(lags[bi]),
            "corr_at_best_lag": float(mean[bi]),
            "frac_trials_positive_corr_lag0": float(np.nanmean(per_trial[:, zero_i] > 0)),
            "median_trial_corr_lag0": float(np.nanmedian(per_trial[:, zero_i])),
        }
        curves[key] = (lags, mean)
    # pooled (all trials concatenated, per-trial demeaned) per-channel correlation
    X, Y = [], []
    for t in trials:
        s = np.stack([box_smooth(c, smooth) for c in t["spikes"]], axis=1)  # T x C
        X.append(s - s.mean(0))
        v = t["vel"]
        Y.append(np.column_stack([v[:, 0] - v[:, 0].mean(), v[:, 1] - v[:, 1].mean(),
                                  speed(v) - speed(v).mean()]))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    if len(X) > 400000:
        idx = np.random.default_rng(0).choice(len(X), 400000, replace=False)
        X, Y = X[idx], Y[idx]
    sx = X.std(0)
    ok = sx > 1e-9
    chan = {}
    for j, nm in enumerate(("vx", "vy", "speed")):
        sy = Y[:, j].std()
        if sy < 1e-9 or not ok.any():
            chan[nm] = None
            continue
        r = np.full(X.shape[1], np.nan)
        r[ok] = (X[:, ok] * Y[:, [j]]).mean(0) / (sx[ok] * sy)
        chan[nm] = r
        out[f"channel_corr_{nm}"] = {
            "median_abs": float(np.nanmedian(np.abs(r))),
            "max_abs": float(np.nanmax(np.abs(r))),
            "n_channels_abs_gt_0.1": int(np.nansum(np.abs(r) > 0.1)),
            "top5_channels": [int(i) for i in np.argsort(-np.nan_to_num(np.abs(r)))[:5]],
        }
    return out, curves, chan


def ema_features(spikes, alphas=(0.5, 0.1, 0.02)):
    """Causal exponential smoothing of every channel at several time constants. -> T x (C*len)."""
    C, T = spikes.shape
    feats = []
    for a in alphas:
        y = np.zeros((C, T), np.float32)
        acc = np.zeros(C, np.float32)
        for t in range(T):
            acc = (1 - a) * acc + a * spikes[:, t]
            y[:, t] = acc
        feats.append(y)
    return np.concatenate(feats, axis=0).T


def ridge_ceiling(train, test):
    """Causal linear decoder on EMA features: does ANY linear readout of this firing track the hand?"""
    if not train:
        return {}
    ev = test if test else None
    tr = train
    note = ""
    if not ev:  # hold out the last 25 % of train trials
        k = max(1, int(0.75 * len(train)))
        tr, ev = train[:k], train[k:]
        note = "no test split -> evaluated on last 25% of train trials"
        if not ev:
            return {}
    Xtr = np.concatenate([ema_features(t["spikes"]) for t in tr])
    Ytr = np.concatenate([t["vel"] for t in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym = Ytr.mean(0)
    Xs = (Xtr - mu) / sd
    # pick ridge lambda on the tail of the training trials
    ntr = len(tr)
    cut = max(1, int(0.8 * ntr))
    sizes = np.cumsum([len(t["vel"]) for t in tr])
    split = int(sizes[cut - 1]) if cut < ntr else int(0.8 * len(Xs))
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        A = Xs[:split]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr[:split] - ym))
        err = np.mean((Xs[split:] @ W + ym - Ytr[split:]) ** 2) if split < len(Xs) else 0
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Ytr - ym))
    losses, P, G = [], [], []
    base = []
    for t in ev:
        p = ((ema_features(t["spikes"]) - mu) / sd) @ W + ym
        losses.append(trial_loss(p, t["vel"]))
        base.append(trial_loss(np.tile(ym, (len(t["vel"]), 1)), t["vel"]))
        P.append(p)
        G.append(t["vel"])
    P, G = np.concatenate(P), np.concatenate(G)
    return {
        "lambda": lam,
        "note": note,
        "ridge_test_loss(mean per-trial sqrtMSE)": float(np.mean(losses)),
        "train_mean_predictor_loss": float(np.mean(base)),
        "ridge_CC_vx": pearson(P[:, 0], G[:, 0]),
        "ridge_CC_vy": pearson(P[:, 1], G[:, 1]),
        "ridge_improvement_over_mean_predictor": float(1 - np.mean(losses) / max(np.mean(base), 1e-9)),
    }


def aligned_averages(trials, smooth, onset_frac, W):
    """Trial averages of summed rate, speed, vx, vy under four alignments."""
    n = len(trials)
    lens = np.array([len(t["vel"]) for t in trials])
    Lmax = int(np.median(lens))
    Lmax = max(Lmax, 10)
    sig = {"rate": [], "speed": [], "vx": [], "vy": []}
    prepared = []
    for t in trials:
        prepared.append({
            "rate": summed_rate(t, smooth),
            "speed": speed(t["vel"]),
            "vx": t["vel"][:, 0],
            "vy": t["vel"][:, 1],
        })
    res = {}

    def stack(extract, L):
        out = {k: np.full((n, L), np.nan) for k in sig}
        for i, p in enumerate(prepared):
            for k in sig:
                seg = extract(i, p[k])
                if seg is None:
                    continue
                out[k][i, : len(seg)] = seg
        return out

    # start-aligned
    res["start"] = (np.arange(Lmax), stack(lambda i, x: x[:Lmax], Lmax))
    # end-aligned: x axis -Lmax..-1
    def end_ex(i, x):
        seg = x[-Lmax:]
        return np.concatenate([np.full(Lmax - len(seg), np.nan), seg])
    res["end"] = (np.arange(-Lmax, 0), stack(end_ex, Lmax))

    # event-aligned
    onsets, peaks = [], []
    for p in prepared:
        sp = p["speed"]
        pk = int(np.argmax(sp))
        thr = onset_frac * sp[pk]
        above = np.where(sp > thr)[0]
        onsets.append(int(above[0]) if len(above) else pk)
        peaks.append(pk)
    xs = np.arange(-W, W + 1)

    def event_stack(ev):
        out = {k: np.full((n, len(xs)), np.nan) for k in sig}
        for i, p in enumerate(prepared):
            T = len(p["speed"])
            lo = max(0, ev[i] - W)
            hi = min(T, ev[i] + W + 1)
            off = lo - (ev[i] - W)
            for k in sig:
                out[k][i, off: off + (hi - lo)] = p[k][lo:hi]
        return out

    res["onset"] = (xs, event_stack(onsets))
    res["peak"] = (xs, event_stack(peaks))
    info = {
        "median_trial_length": int(np.median(lens)),
        "onset_index_percentiles": pct(onsets),
        "peak_speed_index_percentiles": pct(peaks),
        "onset_rule": f"first sample where speed > {onset_frac:.2f} * trial peak speed",
    }
    return res, info


def aligned_coupling(res):
    """How well does trial-averaged firing track trial-averaged speed? (per alignment)"""
    out = {}
    for name, (x, d) in res.items():
        cnt = np.sum(np.isfinite(d["speed"]), axis=0)
        ok = cnt >= max(3, int(0.3 * d["speed"].shape[0]))
        with np.errstate(all="ignore"):
            r = np.nanmean(d["rate"], axis=0)[ok]
            s = np.nanmean(d["speed"], axis=0)[ok]
        out[name] = {"corr_avg_rate_vs_avg_speed": pearson(r, s) if ok.sum() > 5 else None,
                     "avg_speed_peak_over_mean": float(np.nanmax(s) / max(np.nanmean(s), 1e-9)) if ok.sum() > 5 else None}
    return out


# --------------------------------------------------------------------------- plots
def plot_aggregate(session, res, info, ntr, out_path, dt_label):
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharey="row")
    titles = {"start": "aligned at trial START", "end": "aligned at trial END",
              "onset": "aligned at movement ONSET", "peak": "aligned at PEAK speed"}
    for j, name in enumerate(("start", "end", "onset", "peak")):
        x, d = res[name]
        mc = max(3, int(0.3 * ntr))
        rm, rs, cnt = nanmean_sem(d["rate"], mc)
        sm, ss, _ = nanmean_sem(d["speed"], mc)
        vxm, vxs, _ = nanmean_sem(d["vx"], mc)
        vym, vys, _ = nanmean_sem(d["vy"], mc)
        a = axes[0, j]
        a.plot(x, rm, color="tab:blue", label="summed firing (all ch.)")
        a.fill_between(x, rm - rs, rm + rs, color="tab:blue", alpha=0.25)
        a.set_ylabel("summed spikes / sample", color="tab:blue")
        a2 = a.twinx()
        a2.plot(x, sm, color="tab:red", label="speed")
        a2.fill_between(x, sm - ss, sm + ss, color="tab:red", alpha=0.25)
        a2.set_ylabel("speed", color="tab:red")
        a.set_title(titles[name])
        a.grid(alpha=0.3)
        if name in ("onset", "peak"):
            a.axvline(0, color="k", lw=0.8, ls="--")
        b = axes[1, j]
        b.plot(x, vxm, label="vx")
        b.fill_between(x, vxm - vxs, vxm + vxs, alpha=0.25)
        b.plot(x, vym, label="vy")
        b.fill_between(x, vym - vys, vym + vys, alpha=0.25)
        b.axhline(0, color="k", lw=0.6)
        b.set_xlabel(f"samples{dt_label}")
        b.set_ylabel("mean velocity")
        b.grid(alpha=0.3)
        if name in ("onset", "peak"):
            b.axvline(0, color="k", lw=0.8, ls="--")
        if j == 0:
            b.legend()
    fig.suptitle(f"{session}: trial-averaged neural activity vs hand velocity  (n={ntr} trials; "
                 f"shaded = ±SEM; points with <30% of trials hidden)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_census(session, train, test, curves, chan, v_lo, v_hi, out_path):
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    for tag, tr, c in (("train", train, "tab:blue"), ("test", test, "tab:orange")):
        if tr:
            ax[0, 0].hist([len(t["vel"]) for t in tr], bins=40, alpha=0.6, color=c, label=tag)
    ax[0, 0].set_title("trial length (samples)")
    ax[0, 0].legend()
    for tag, tr, c in (("train", train, "tab:blue"), ("test", test, "tab:orange")):
        if tr:
            V = np.concatenate([t["vel"] for t in tr])
            lim = np.percentile(np.abs(V), 99.9)
            ax[0, 1].hist(V[:, 0], bins=200, range=(-lim, lim), alpha=0.5, color=c, label=f"{tag} vx", density=True)
    ax[0, 1].set_yscale("log")
    if v_lo is not None:
        ax[0, 1].axvline(v_lo, color="k", ls="--")
        ax[0, 1].axvline(v_hi, color="k", ls="--", label="scaler bounds")
    ax[0, 1].set_title("vx distribution (log density)")
    ax[0, 1].legend()
    for tr, c, lab in ((train, "tab:blue", "train"), (test, "tab:orange", "test")):
        if tr:
            m = np.array([t["vel"].mean(0) for t in tr])
            ax[0, 2].scatter(m[:, 0], m[:, 1], s=8, alpha=0.5, color=c, label=lab)
    ax[0, 2].set_title("per-trial MEAN velocity (vx, vy)")
    ax[0, 2].legend()
    ax[0, 2].axhline(0, color="k", lw=0.5)
    ax[0, 2].axvline(0, color="k", lw=0.5)
    for key, col in (("speed", "k"), ("vx", "tab:blue"), ("vy", "tab:orange")):
        lags, m = curves[key]
        ax[1, 0].plot(lags, m, color=col, label=key)
    ax[1, 0].axvline(0, color="gray", lw=0.6)
    ax[1, 0].axhline(0, color="gray", lw=0.6)
    ax[1, 0].set_title("lagged corr: summed firing[t] vs hand[t+lag]  (lag>0 = neural leads)")
    ax[1, 0].set_xlabel("lag (samples)")
    ax[1, 0].legend()
    for nm, col in (("vx", "tab:blue"), ("vy", "tab:orange"), ("speed", "k")):
        r = chan.get(nm)
        if r is not None:
            ax[1, 1].hist(r[np.isfinite(r)], bins=40, alpha=0.5, color=col, label=nm)
    ax[1, 1].set_title("per-channel corr with hand (per-trial demeaned, smoothed)")
    ax[1, 1].legend()
    # firing statistics
    ch_rate = sum(t["spikes"].sum(1) for t in train) / sum(t["spikes"].shape[1] for t in train)
    ax[1, 2].bar(np.arange(len(ch_rate)), ch_rate)
    ax[1, 2].set_title("per-channel mean rate (spikes/sample, train)")
    ax[1, 2].set_xlabel("channel")
    fig.suptitle(session)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_examples(session, trials, smooth, out_path, n=4, seed=0):
    if not trials:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(trials), size=min(n, len(trials)), replace=False)
    fig, axes = plt.subplots(len(idx), 1, figsize=(14, 2.8 * len(idx)), squeeze=False)
    for a, i in zip(axes[:, 0], idx):
        t = trials[i]
        r = summed_rate(t, smooth)
        a.plot(r, color="tab:blue", label="summed firing")
        a.set_ylabel("spikes/sample", color="tab:blue")
        a2 = a.twinx()
        a2.plot(t["vel"][:, 0], color="tab:red", lw=0.9, label="vx")
        a2.plot(t["vel"][:, 1], color="tab:green", lw=0.9, label="vy")
        a2.set_ylabel("velocity")
        a.set_title(f"{t['name']}  (T={len(r)})")
    fig.suptitle(f"{session}: random example trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_summary(all_res, out_path):
    names = list(all_res)
    short = [n.replace("sub-", "").replace("_behavior+ecephys", "") for n in names]
    def g(fn, default=np.nan):
        out = []
        for n in names:
            try:
                v = fn(all_res[n])
                out.append(np.nan if v is None else v)
            except Exception:  # noqa: BLE001
                out.append(default)
        return np.array(out, dtype=float)
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    ax = ax.ravel()
    ax[0].bar(short, g(lambda r: r["velocity"]["trivial_predictor_loss_on_test"]["predict_train_mean"]
                      if "trivial_predictor_loss_on_test" in r["velocity"] else
                      r["velocity"]["trivial_predictor_loss_on_train(no test split)"]["predict_train_mean"]))
    ax[0].set_title("loss of predicting the TRAIN MEAN (target-scale floor)")
    ax[1].bar(short, g(lambda r: r["ridge"]["ridge_test_loss(mean per-trial sqrtMSE)"]))
    ax[1].set_title("causal ridge-regression test loss")
    ax[2].bar(short, g(lambda r: r["ridge"]["ridge_improvement_over_mean_predictor"]))
    ax[2].set_title("ridge improvement over mean predictor (0 = no info)")
    ax[3].bar(short, g(lambda r: r["firing_hand"]["speed"]["corr_lag0_mean_over_trials"]))
    ax[3].set_title("corr(summed firing, speed) lag 0")
    ax[4].bar(short, g(lambda r: r["velocity"]["train"]["speed_percentiles"]["p99"]))
    ax[4].set_title("speed p99 (train)")
    ax[5].bar(short, g(lambda r: r["velocity"]["train"]["abs_max"]))
    ax[5].set_title("max |velocity| (train)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- warnings / report
def make_warnings(r, min_win):
    w = []
    c = r["train_census"]
    if c.get("n_trials", 0) == 0:
        return ["NO TRAIN TRIALS LOADED"]
    if c["frac_trials_len_le_%d" % min_win] > 0.05:
        w.append(f"{100*c['frac_trials_len_le_%d' % min_win]:.1f}% of train trials are <= {min_win} samples "
                 f"(too short for ANN windows; tiny contribution for SNN).")
    if c["n_empty_trials(no spikes)"] > 0:
        w.append(f"{c['n_empty_trials(no spikes)']} train trial(s) contain ZERO spikes.")
    if c["frac_dead_channels"] > 0.1:
        w.append(f"{100*c['frac_dead_channels']:.0f}% of channels never spike in train.")
    if c["n_nonfinite_values"] > 0:
        w.append(f"{c['n_nonfinite_values']} NaN/inf values in train data.")
    v = r["velocity"].get("train", {})
    triv = [k for k in r["velocity"] if k.startswith("trivial")]
    if triv:
        t0 = r["velocity"][triv[0]]
        if t0["predict_train_mean"] > 150:
            w.append(f"Predicting the train mean already gives loss {t0['predict_train_mean']:.0f}: the TARGET SCALE "
                     f"itself is huge. A loss of 'hundreds' may just be the velocity magnitude, not a decoder failure.")
        if t0["predict_per_trial_mean(oracle)"] < 0.5 * t0["predict_train_mean"]:
            w.append("Knowing only each trial's mean velocity would cut the loss by >50%: velocity is dominated by a "
                     "trial-constant offset (between-trial variance).")
    if v:
        bv = v["between_trial_var_fraction(vx,vy)"]
        if max(bv) > 0.5:
            w.append(f"Between-trial variance fraction of velocity = {bv[0]:.2f}/{bv[1]:.2f} (vx/vy): most variance is "
                     f"a per-trial constant, which a spike decoder with state reset per trial cannot recover.")
        if v["abs_max"] > 10 * v["speed_percentiles"].get("p99", 1e9):
            w.append(f"Heavy outliers: max |v| = {v['abs_max']:.0f} vs speed p99 = {v['speed_percentiles']['p99']:.0f}.")
        if v.get("lag1_autocorr_median(smoothness)") is not None and v["lag1_autocorr_median(smoothness)"] < 0.5:
            w.append("Velocity is very NOISY sample-to-sample (lag-1 autocorr < 0.5); looks unsmoothed/derivative-like.")
        if v["n_zero_variance_trials"] > 0:
            w.append(f"{v['n_zero_variance_trials']} trial(s) have constant velocity.")
        if "frac_samples_outside_[v_lo,v_hi]" in v and v["frac_samples_outside_[v_lo,v_hi]"] > 0.01:
            w.append(f"{100*v['frac_samples_outside_[v_lo,v_hi]']:.1f}% of velocity samples fall outside the scaler bounds "
                     f"(SNN output is bounded there -> irreducible error "
                     f"{v['loss_floor_from_clipping_to_bounds(mean of per-trial sqrtMSE)']:.1f}).")
    sh = r["velocity"].get("train_test_mean_shift_in_train_std_units")
    if sh and max(abs(x) for x in sh) > 0.5:
        w.append(f"Train->test mean velocity shift of {sh[0]:.2f}/{sh[1]:.2f} train-std (vx/vy): distribution shift.")
    rd = r["ridge"]
    if rd:
        spc = r["firing_hand"].get("speed", {}).get("corr_at_best_lag")
        if rd["ridge_improvement_over_mean_predictor"] < 0.05 and spc is not None and abs(spc) > 0.3:
            w.append(f"Firing tracks SPEED magnitude (r={spc:.2f}) but a linear decoder does not recover signed "
                     "vx/vy: direction information is weak or varies across trials.")
        elif rd["ridge_improvement_over_mean_predictor"] < 0.05:
            w.append("A causal linear decoder on the firing barely beats the mean predictor "
                     f"({100*rd['ridge_improvement_over_mean_predictor']:.1f}% better): little linear velocity "
                     "information in these spikes, or spikes and velocity are misaligned.")
        else:
            w.append(f"INFO exists: ridge decoder improves loss by {100*rd['ridge_improvement_over_mean_predictor']:.0f}% "
                     f"(CC vx/vy = {rd['ridge_CC_vx']:.2f}/{rd['ridge_CC_vy']:.2f}); if the SNN is stuck, suspect "
                     "scaling/optimisation rather than the data.")
    fh = r["firing_hand"]
    sp = fh.get("speed", {})
    if sp.get("corr_at_best_lag") is not None and abs(sp["corr_at_best_lag"]) < 0.1:
        w.append("Summed firing vs speed correlation < 0.1 at every lag: no gross firing-movement coupling.")
    elif sp.get("best_lag_samples") not in (None,) and abs(sp.get("best_lag_samples", 0)) > 5:
        w.append(f"Firing-speed correlation peaks at lag {sp['best_lag_samples']} samples (not ~0): check alignment.")
    return w


def write_report(all_res, path):
    L = []
    for sess, r in all_res.items():
        L.append("=" * 100)
        L.append(sess)
        L.append("=" * 100)
        c, ct = r["train_census"], r.get("test_census", {})
        L.append(f"train trials {c.get('n_trials')}, test trials {ct.get('n_trials', 0)}, channels {c.get('n_channels')}")
        if c.get("n_trials"):
            lp = c["length_percentiles"]
            L.append(f"train length p0/p50/p100 = {lp['p0']:.0f}/{lp['p50']:.0f}/{lp['p100']:.0f}")
        for tag in ("train", "test"):
            v = r.get("velocity", {}).get(tag)
            if v:
                L.append(f"{tag}: vel mean {np.round(v['mean'], 1).tolist()}  std {np.round(v['std'], 1).tolist()}  "
                         f"speed p50/p99 {v['speed_percentiles']['p50']:.1f}/{v['speed_percentiles']['p99']:.1f}  "
                         f"abs max {v['abs_max']:.1f}")
        for k, v in r.get("velocity", {}).items():
            if k.startswith("trivial"):
                L.append(f"{k}: " + ", ".join(f"{a}={b:.1f}" for a, b in v.items() if isinstance(b, float)))
        if r.get("ridge"):
            L.append("ridge: " + ", ".join(f"{a}={b:.3f}" for a, b in r["ridge"].items() if isinstance(b, float)))
        for key in ("speed", "vx", "vy"):
            x = r.get("firing_hand", {}).get(key, {})
            if x.get("corr_at_best_lag") is not None:
                L.append(f"firing vs {key}: lag0 r={x['corr_lag0_mean_over_trials']:.3f}  "
                         f"best lag {x['best_lag_samples']} r={x['corr_at_best_lag']:.3f}")
        for name, x in r.get("aligned_coupling", {}).items():
            L.append(f"aligned[{name}]: corr(avg rate, avg speed)={x['corr_avg_rate_vs_avg_speed']}")
        L.append("WARNINGS / FINDINGS:")
        for w in r.get("warnings", []):
            L.append("  - " + w)
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def diagnose_session(sess_dir, out_dir, args):
    name = os.path.basename(sess_dir.rstrip("/"))
    paths = session_paths(out_dir, name)
    train = load_split(os.path.join(sess_dir, "train"))
    test = load_split(os.path.join(sess_dir, "test"))
    print(f"\n[{name}] train={len(train)} trials, test={len(test)} trials")
    if not train:
        r = {"train_census": {"n_trials": 0}, "warnings": ["NO TRAIN TRIALS LOADED"]}
        atomic_json_dump({"params": current_params(args), "result": r}, paths["json"])
        return name, r
    r = {}
    r["train_census"] = trial_census(train, args.min_window)
    r["test_census"] = trial_census(test, args.min_window)
    r["velocity"] = velocity_census(train, test, args.v_lo, args.v_hi, args.min_window)
    ev_trials = train + test
    fh, curves, chan = firing_hand_correlation(ev_trials, args.smooth, args.max_lag)
    r["firing_hand"] = fh
    r["firing_hand_train_only"] = firing_hand_correlation(train, args.smooth, args.max_lag)[0]
    r["ridge"] = ridge_ceiling(train, test)
    agg, info = aligned_averages(ev_trials, args.smooth, args.onset_frac, args.window)
    r["aligned_info"] = info
    r["aligned_coupling"] = aligned_coupling(agg)
    r["warnings"] = make_warnings(r, args.min_window)
    dt = f" (x {args.dt_ms:g} ms)" if args.dt_ms else ""
    plot_aggregate(name, agg, info, len(ev_trials), paths["aggregate"], dt)
    plot_census(name, train, test, curves, chan, args.v_lo, args.v_hi, paths["census"])
    plot_examples(name, ev_trials, args.smooth, paths["examples"])
    # Round-trip through JSON so fresh and cached results have identical types downstream.
    r = json.loads(json.dumps(r, default=_json_default))
    # Written LAST: its presence marks the session as complete.
    atomic_json_dump({"params": current_params(args), "result": r}, paths["json"])
    return name, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dir containing one subdir per session (each with train/ test/)")
    ap.add_argument("--out-dir", default="hkm_diagnostics")
    ap.add_argument("--sessions", nargs="*", help="only these session dir names (default: all)")
    ap.add_argument("--smooth", type=int, default=5, help="box-smoothing width (samples) for summed firing")
    ap.add_argument("--max-lag", type=int, default=150, help="max |lag| (samples) in lagged cross-correlation")
    ap.add_argument("--window", type=int, default=100, help="half-window (samples) for onset/peak alignment")
    ap.add_argument("--onset-frac", type=float, default=0.2, help="onset = first sample with speed > frac*peak speed")
    ap.add_argument("--min-window", type=int, default=65, help="ANN window length; trials <= this give no ANN rows")
    ap.add_argument("--dt-ms", type=float, default=None, help="optional ms per sample, for axis labels only")
    ap.add_argument("--v-lo", type=float, default=None, help="velocity scaler lower bound used by the trainer")
    ap.add_argument("--v-hi", type=float, default=None, help="velocity scaler upper bound used by the trainer")
    ap.add_argument("--force", action="store_true",
                    help="recompute every session even if results already exist in --out-dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sess_dirs = sorted(d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d))
    if args.sessions:
        sess_dirs = [d for d in sess_dirs if os.path.basename(d) in set(args.sessions)]
    if not sess_dirs:
        sys.exit(f"no session directories under {args.data_root}")
    if (args.v_lo is None) != (args.v_hi is None):
        sys.exit("pass both --v-lo and --v-hi, or neither")

    legacy = {} if args.force else load_legacy_combined(args.out_dir)

    all_res, n_skipped, n_done, failed = {}, 0, 0, []
    for i, d in enumerate(sess_dirs, 1):
        name = os.path.basename(d.rstrip("/"))
        if not args.force:
            cached, reason = load_completed(name, args.out_dir, args, legacy)
            if cached is not None:
                print(f"[{i}/{len(sess_dirs)}] {name}: skipping ({reason})")
                all_res[name] = cached
                n_skipped += 1
                continue
            if reason != "not done":
                print(f"[{i}/{len(sess_dirs)}] {name}: recomputing ({reason})")
        t0 = time.time()
        try:
            n, r = diagnose_session(d, args.out_dir, args)
        except Exception:  # noqa: BLE001
            print(f"[{i}/{len(sess_dirs)}] {name}: FAILED\n{traceback.format_exc()}")
            failed.append(name)
            plt.close("all")
            continue
        all_res[n] = r
        n_done += 1
        print(f"[{i}/{len(sess_dirs)}] {name}: done in {time.time() - t0:.0f}s")

    print(f"\nSessions: {n_done} computed, {n_skipped} skipped (already done), {len(failed)} failed")

    good = {k: v for k, v in all_res.items() if "velocity" in v}
    if good:
        plot_summary(good, os.path.join(args.out_dir, "summary_all_sessions.png"))
    atomic_json_dump(all_res, os.path.join(args.out_dir, "diagnostics.json"))
    print(write_report(all_res, os.path.join(args.out_dir, "diagnostics.txt")))
    print(f"\nWrote figures + diagnostics.json/.txt to {args.out_dir}")
    if failed:
        print("FAILED sessions (rerun the same command to retry them): " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()