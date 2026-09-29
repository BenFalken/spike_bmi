"""
Aggregates the new, minimal speck_results.json files (one per session, written by the current
infer_snn_speck.py -- torch and speck only, RMSE/CC/latency/power, no discretized twin, no
input/power-timestamp audits) into one aggregated_summary.json plus a printed summary table.

Replaces the earlier, much larger aggregate_speck_results.py, which read fields (rmse_axes,
per-implementation discretized entries, speck_input_audit, power_audit's diagnostic sub-fields)
that the current infer_snn_speck.py no longer writes. If you still have speck_results.json files
from that earlier script, this will not read them -- re-run the sessions with the current
infer_snn_speck.py first.

Key names in aggregated_summary.json's impls[impl] are kept IDENTICAL to the previous
aggregator's output (rmse_vs_gt, cc_x_vs_gt, cc_y_vs_gt, cc_avg_vs_gt, latency_per_timestep_ms,
dynapcnn_param_count) specifically so decoder_comparison_4x2.py and
plot_decoder_efficiency_aggregate.py keep working completely unmodified -- verified end to end
against a real run of both scripts. rmse_vs_gt is the Oscar/sklearn 'mean of axes' convention
(comparable to test_all_decoders.py's decoders); rmse_pooled_vs_gt (train_bmi.py's own
convention) is kept alongside it.

Usage:
    python aggregate_speck_results.py --results-root ./speck_results/bmi/indy --output-dir ./speck_summary/bmi/indy
"""

import argparse
import glob
import json
import math
import os


def discover_sessions(results_root):
    paths = sorted(glob.glob(os.path.join(results_root, "*", "speck_results.json")))
    if not paths:
        raise FileNotFoundError(
            f"No speck_results.json found under {results_root}/*/ -- has infer_snn_speck.py "
            f"actually finished any sessions yet? (If these are from the OLD, larger "
            f"infer_snn_speck.py, re-run with the current one -- this reads a different, "
            f"smaller schema, see this file's own module docstring.)")
    out = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        out[d.get("session_id", os.path.basename(os.path.dirname(p)))] = d
    return out


def _summarize(values):
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "values": values}


def aggregate(session_metrics):
    impls_present = sorted({impl for m in session_metrics.values() for impl in m.get("active_impls", [])})
    wait_times = sorted({m["speck_wait_time_s"] for m in session_metrics.values()
                         if m.get("speck_wait_time_s") is not None})
    out = {"n_sessions": len(session_metrics), "impls": {}, "speck_wait_time_s_recorded": wait_times}
    if len(wait_times) > 1:
        out["speck_wait_time_warning"] = (f"sessions used DIFFERENT chip readout wait times "
                                          f"{wait_times} -- their speck results are not comparable")

    for impl in impls_present:
        sessions, dropped = [], []
        rmse_mean_of_axes, rmse_pooled, cc_x, cc_y, cc_avg, latency_ms, power_mw = [], [], [], [], [], [], []
        energy_methods_seen = {}  # method -> n_sessions -- infer_snn_speck.py's own per-session
        # "energy_method" was never previously read here at all, despite being saved in every
        # session's speck_results.json -- meaning nothing downstream of this aggregator could
        # ever tell a real RAPL/chip measurement apart from torch's psutil-proxy fallback
        # without re-opening individual session files by hand. Tracked the same way
        # plot_decoder_efficiency_aggregate.py's load_profiles() already tracks it for the ANN
        # decoders: majority wins, and a genuine cross-session split (this laptop having RAPL
        # access change between runs, say) is flagged rather than silently averaged over.
        for session_id, m in session_metrics.items():
            r = m.get("results", {}).get(impl)
            if r is None:
                continue
            missing = [k for k in ("rmse_x", "rmse_y", "cc_x", "cc_y", "latency_ms_per_step", "power_mw") if r.get(k) is None]
            if missing:
                dropped.append({"session": session_id, "missing": missing})
                continue
            sessions.append(session_id)
            rmse_mean_of_axes.append((r["rmse_x"] + r["rmse_y"]) / 2)
            rmse_pooled.append(r["rmse_pooled"])
            cc_x.append(r["cc_x"]); cc_y.append(r["cc_y"]); cc_avg.append((r["cc_x"] + r["cc_y"]) / 2)
            latency_ms.append(r["latency_ms_per_step"]); power_mw.append(r["power_mw"])
            if r.get("energy_method") is not None:
                energy_methods_seen[r["energy_method"]] = energy_methods_seen.get(r["energy_method"], 0) + 1
        if dropped:
            print(f"  WARNING [{impl}]: {len(dropped)} session(s) dropped (missing fields): "
                  + "; ".join(f"{d['session']} ({', '.join(d['missing'])})" for d in dropped))
        if len(energy_methods_seen) > 1:
            print(f"  WARNING [{impl}]: energy_method is NOT consistent across sessions: "
                  f"{energy_methods_seen} -- some sessions measured real energy, others used a "
                  f"proxy estimate (or vice versa). The majority method is recorded below, but "
                  f"averaging these together mixes a measurement with an estimate.")
        energy_method = (max(energy_methods_seen, key=energy_methods_seen.get)
                         if energy_methods_seen else None)
        out["impls"][impl] = {
            "n_sessions_with_impl": len(sessions), "sessions": sessions, "dropped_sessions": dropped,
            "latency_per_timestep_ms": _summarize(latency_ms),
            "power_mw": _summarize(power_mw),
            "energy_method": energy_method,
            "rmse_vs_gt": _summarize(rmse_mean_of_axes),          # Oscar/sklearn convention -- what the
            "rmse_mean_of_axes_vs_gt": _summarize(rmse_mean_of_axes),  # 4x2/efficiency plots read
            "rmse_pooled_vs_gt": _summarize(rmse_pooled),
            "cc_x_vs_gt": _summarize(cc_x), "cc_y_vs_gt": _summarize(cc_y), "cc_avg_vs_gt": _summarize(cc_avg),
        }

    param_counts = {m["dynapcnn_param_count"] for m in session_metrics.values()
                    if m.get("dynapcnn_param_count") is not None}
    if len(param_counts) == 1:
        out["dynapcnn_param_count"] = param_counts.pop()
    elif len(param_counts) > 1:
        out["dynapcnn_param_count"] = None
        out["dynapcnn_param_count_warning"] = f"sessions disagree on dynapcnn_param_count: {sorted(param_counts)}"
    else:
        out["dynapcnn_param_count"] = None
    return out


def print_table(aggregated, label):
    print(f"\n{'=' * 72}\n{label} -- {aggregated['n_sessions']} session(s) aggregated\n{'=' * 72}")
    if aggregated.get("dynapcnn_param_count") is not None:
        print(f"  DynapcnnNetwork param count: {aggregated['dynapcnn_param_count']:,}")
    if aggregated.get("speck_wait_time_s_recorded"):
        print(f"  Chip readout wait: {[f'{w * 1e3:g} ms' for w in aggregated['speck_wait_time_s_recorded']]}")
    if aggregated.get("speck_wait_time_warning"):
        print(f"  WARNING: {aggregated['speck_wait_time_warning']}")
    print(f"  {'Impl':<8} {'n':>3} {'latency (ms/step)':>20} {'power (mW)':>18} "
          f"{'RMSE, Oscar conv.':>20} {'CC':>16}")
    for impl, d in aggregated["impls"].items():
        lat, pw, rmse, cc = d["latency_per_timestep_ms"], d["power_mw"], d["rmse_vs_gt"], d["cc_avg_vs_gt"]

        def fmt(s):
            return f"{s['mean']:.4f} +/- {s['std']:.4f}" if s["n"] else "n/a"
        method_tag = {"rapl": "real RAPL", "chip_power_monitor": "real chip",
                      "proxy_psutil": "PROXY ESTIMATE"}.get(d.get("energy_method"), d.get("energy_method") or "n/a")
        print(f"  {impl:<8} {d['n_sessions_with_impl']:>3} {fmt(lat):>20} {fmt(pw):>18} "
              f"{fmt(rmse):>20} {fmt(cc):>16}   power method: {method_tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    session_metrics = discover_sessions(args.results_root)
    aggregated = aggregate(session_metrics)
    label = args.label or os.path.basename(os.path.normpath(args.results_root))
    print_table(aggregated, label)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "aggregated_summary.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
