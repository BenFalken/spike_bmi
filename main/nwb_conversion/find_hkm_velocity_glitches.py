#!/usr/bin/env python3
"""
Locates non-physical velocity spikes in the EXISTING HKM whole-trial .pkl files and describes
their shape, so you can see what the glitches are before/after rebuilding with the de-spike step
(convert_nwb_trials_to_raw_h5.py --max-speed).

For every session under --data-root (each with train/ and test/ N.pkl files holding
`velocity` T x 2 in physical units) it reports:
  * how many samples/trials exceed --threshold (speed = |v|), and the glitch magnitude spread
  * a speed histogram across [500 ... inf) -- a wide EMPTY band between real motion (~<1500) and
    the glitches shows that any threshold inside the band cleans identically
  * where in the trial they sit: first sample / last sample / interior
  * their shape: isolated 1-sample spike vs longer run; whether the spike is followed by an
    opposite-sign spike (out-and-back tracker jump) or not (step)
  * the 10 largest examples (file, sample index, trial length, speed, neighbouring speeds)

Usage:
  python find_hkm_velocity_glitches.py \
      --data-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/hkm/nitschke/mua \
      --out-dir hkm_diagnostics/nitschke_glitches [--threshold 3000]
Needs only numpy.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

BINS = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, 30000, 1e12]


def load_vel(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    v = d["velocity"]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 2 and v.shape[0] == 2 and v.shape[1] != 2:
        v = v.T
    return v


def runs_of(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(g[0], g[-1]) for g in np.split(idx, splits)]


def analyse_session(sess_dir, thr):
    out = {"hist": np.zeros(len(BINS) - 1, dtype=np.int64), "n_samples": 0, "n_trials": 0,
           "trials_with_glitch": 0, "glitch_samples": 0, "runs": [], "examples": [],
           "vel": {"train": [], "test": []}}
    for split in ("train", "test"):
        for path in sorted(glob.glob(os.path.join(sess_dir, split, "*.pkl"))):
            try:
                v = load_vel(path)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {path}: {e}")
                continue
            out["vel"][split].append(v)
            sp = np.linalg.norm(v, axis=1)
            T = len(sp)
            out["n_trials"] += 1
            out["n_samples"] += T
            out["hist"] += np.histogram(sp, bins=BINS)[0]
            bad = sp > thr
            if not bad.any():
                continue
            out["trials_with_glitch"] += 1
            out["glitch_samples"] += int(bad.sum())
            for a, b in runs_of(bad):
                length = b - a + 1
                where = "first" if a == 0 else ("last" if b == T - 1 else "interior")
                # opposite-sign follow-up: next big sample after this run has opposite direction
                follow = None
                nxt = np.flatnonzero(sp[b + 1:b + 6] > thr)
                if nxt.size:
                    j = b + 1 + nxt[0]
                    follow = "out-and-back" if float(np.dot(v[a], v[j])) < 0 else "same-direction"
                else:
                    follow = "no follow-up (step/one-sided)"
                out["runs"].append({"len": int(length), "where": where, "follow": follow})
                k = a + int(np.argmax(sp[a:b + 1]))
                out["examples"].append({
                    "file": f"{split}/{os.path.basename(path)}", "idx": int(k), "T": int(T),
                    "speed": float(sp[k]), "prev_speed": float(sp[k - 1]) if k > 0 else None,
                    "next_speed": float(sp[k + 1]) if k + 1 < T else None})
    return out


def baselines(vel):
    """Fast (seconds) stand-ins for the slow diagnostic's key numbers: speed percentiles, train/test std,
    and the RMSE of predicting zero / the train mean on the test split (pooled over both components)."""
    tr = np.concatenate(vel["train"]) if vel["train"] else np.zeros((0, 2))
    te = np.concatenate(vel["test"]) if vel["test"] else np.zeros((0, 2))
    allv = np.concatenate([tr, te]) if len(tr) + len(te) else np.zeros((0, 2))
    if not len(allv):
        return {}
    sp = np.linalg.norm(allv, axis=1)
    b = {"speed_p50": float(np.percentile(sp, 50)), "speed_p99": float(np.percentile(sp, 99)),
         "speed_p99.5": float(np.percentile(sp, 99.5)), "speed_max": float(sp.max()),
         "abs_max_component": float(np.abs(allv).max())}
    if len(tr):
        b["train_std"] = float(tr.std())
    if len(te):
        b["test_std"] = float(te.std())
        b["test_rmse_predict_zero"] = float(np.sqrt((te ** 2).mean()))
        if len(tr):
            b["test_rmse_predict_train_mean"] = float(np.sqrt(((te - tr.mean(axis=0)) ** 2).mean()))
    return b


def _f(x):
    return "   -" if x is None else f"{x:.0f}"


def fmt_hist(h):
    labels = [f"{int(BINS[i])}-{'inf' if BINS[i+1] > 1e11 else int(BINS[i+1])}" for i in range(len(h))]
    return "  ".join(f"[{l}]={int(c)}" for l, c in zip(labels, h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="hkm_glitches")
    ap.add_argument("--threshold", type=float, default=3000.0, help="speed (|v|, units/s) counted as a glitch")
    ap.add_argument("--abs-max", type=float, default=3000.0, help="PASS requires max |component| <= this")
    ap.add_argument("--only-session", default=None, help="only scan the session directory with this exact name")
    ap.add_argument("--fail-on-glitch", action="store_true", help="exit with status 1 if any scanned session is FAIL")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sessions = sorted(d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d))
    if args.only_session:
        sessions = [d for d in sessions if os.path.basename(d) == args.only_session]
    if not sessions:
        sys.exit(f"no session directories under {args.data_root}")

    lines, js, verdicts = [], {}, {}
    for sd in sessions:
        name = os.path.basename(sd)
        print(f"[{name}] scanning ...")
        r = analyse_session(sd, args.threshold)
        runs = r["runs"]
        by_len = {}
        for x in runs:
            by_len[x["len"]] = by_len.get(x["len"], 0) + 1
        loc = {k: sum(1 for x in runs if x["where"] == k) for k in ("first", "interior", "last")}
        fol = {}
        for x in runs:
            fol[x["follow"]] = fol.get(x["follow"], 0) + 1
        ex = sorted(r["examples"], key=lambda e: -e["speed"])
        sp_all = np.array([e["speed"] for e in r["examples"]]) if r["examples"] else np.array([])
        lines += ["=" * 100, name, "=" * 100,
                  f"trials {r['n_trials']}, samples {r['n_samples']}",
                  f"speed histogram: {fmt_hist(r['hist'])}",
                  f"glitch samples (> {args.threshold:g}): {r['glitch_samples']}  in {r['trials_with_glitch']} trial(s) "
                  f"({100 * r['trials_with_glitch'] / max(r['n_trials'], 1):.1f}% of trials)",
                  f"glitch runs: {len(runs)}   run-length histogram {dict(sorted(by_len.items()))}",
                  f"location in trial: {loc}",
                  f"shape: {fol}"]
        if sp_all.size:
            lines.append(f"glitch peak speed: min {sp_all.min():.0f}  median {np.median(sp_all):.0f}  max {sp_all.max():.0f}")
        lines.append("largest examples:")
        for e in ex[:10]:
            lines.append(f"   {e['file']:>14s}  idx {e['idx']:4d}/{e['T']:<4d}  speed {e['speed']:9.0f}  "
                         f"prev {_f(e['prev_speed'])}  next {_f(e['next_speed'])}")
        bl = baselines(r["vel"])
        ok = r["glitch_samples"] == 0 and bl.get("abs_max_component", 0) <= args.abs_max
        verdict = "PASS" if ok else "FAIL"
        verdicts[name] = verdict
        lines.append("baselines (compare before/after; clean session 20090922 was ~131 for predict-zero in the old diagnostic):")
        lines.append("   " + "  ".join(f"{k}={v:.1f}" for k, v in bl.items()))
        lines.append(f"VERDICT {verdict}: glitch samples = {r['glitch_samples']}, abs max component = "
                     f"{bl.get('abs_max_component', float('nan')):.0f} (limit {args.abs_max:g})")
        lines.append("")
        js[name] = bl and {"baselines": bl, "verdict": verdict} or {}
        js[name].update({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items() if k not in ("examples", "vel")})
        js[name]["top_examples"] = ex[:50]
    lines += ["=" * 100, "SUMMARY: " + "  ".join(f"{k}={v}" for k, v in verdicts.items())]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "glitches.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, "glitches.json"), "w") as f:
        json.dump(js, f, indent=1)
    print(f"\nWrote {args.out_dir}/glitches.txt and glitches.json")
    if args.fail_on_glitch and any(v == "FAIL" for v in verdicts.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
