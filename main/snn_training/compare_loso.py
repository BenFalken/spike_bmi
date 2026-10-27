"""
Compare the LOSO fine-tuned LSNN with the LOSO fine-tuned SNN, session by session,
for one subject: each session's best fine-tuned checkpoint of each model on that
session's test split (compare_lsnn.evaluate: velocity RMS error, r, R^2, spike
rates, CPU time per step).

Reads (the pooled scripts' outputs; run_snn_pooled_pretrain.sbatch shows how to
submit them, MODEL=lsnn for the LSNN):
    $DATA_ROOT/snn_checkpoints/<exp>/<subject>/loso_finetuned_binarized_soft/<session>/   SNN
    $DATA_ROOT/snn_checkpoints/<exp>/<subject>/lsnn_loso_finetuned_binarized/<session>/   LSNN
Sessions missing either checkpoint are listed and skipped.

Writes to --out-dir: loso_comparison.json, loso_comparison.md (per session and the
mean, with the number of sessions each model wins), loso_comparison.png.

    python snn_training/compare_loso.py --data-root $DATA_ROOT --experiment bmi --subject indy \\
        --out-dir lsnn_vs_snn_loso/indy --device cpu
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_lsnn import evaluate  # noqa: E402

MODELS = {"SNN (LOSO)": "loso_finetuned_binarized_soft", "LSNN (LOSO)": "lsnn_loso_finetuned_binarized"}
METRICS = [("rmse", "velocity error (RMS)", "{:.1f}", -1), ("r2_mean", "R² (mean of x, y)", "{:.3f}", 1),
           ("r_mean", "r (mean of x, y)", "{:.3f}", 1)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/users/bfalkenb/scratch/bfalkenb/data"))
    p.add_argument("--experiment", default="bmi")
    p.add_argument("--subject", default="indy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.data_root, "snn_checkpoints", args.experiment, args.subject)
    roots = {m: os.path.join(base, d) for m, d in MODELS.items()}
    sessions = sorted(set.intersection(*[
        {s for s in os.listdir(r) if os.path.isfile(os.path.join(r, s, "best_model_weights.pth"))}
        if os.path.isdir(r) else set() for r in roots.values()]))
    missing = sorted({s for r in roots.values() if os.path.isdir(r) for s in os.listdir(r)} - set(sessions))
    print(f"{len(sessions)} sessions with both models" + (f"; skipped (one missing): {missing}" if missing else ""))
    if not sessions:
        sys.exit("nothing to compare: run the LOSO pretrain + fine-tune for both models first")

    results = {}
    for s in sessions:
        results[s] = {}
        for m, r in roots.items():
            res, _, _ = evaluate(os.path.join(r, s), torch.device(args.device))
            results[s][m] = res
        print(s, " | ".join(f"{m}: {results[s][m]['rmse']:.1f}" for m in MODELS), flush=True)
    with open(os.path.join(args.out_dir, "loso_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    names = list(MODELS)
    lines = [f"# LOSO fine-tuned SNN vs LSNN: {args.experiment}/{args.subject}, {len(sessions)} sessions", "",
             "| session | " + " | ".join(f"{m} error" for m in names) + " | " + " | ".join(f"{m} R²" for m in names) + " |",
             "|---|" + "---|" * (2 * len(names))]
    for s in sessions:
        lines.append(f"| {s} | " + " | ".join(f"{results[s][m]['rmse']:.1f}" for m in names) + " | "
                     + " | ".join(f"{results[s][m]['r2_mean']:.3f}" for m in names) + " |")
    lines += ["", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, fmt, sign in METRICS:
        vals = {m: np.array([results[s][m][key] for s in sessions]) for m in names}
        lines.append(f"| mean {label} | " + " | ".join(fmt.format(vals[m].mean()) for m in names) + " |")
        better = sign * (vals[names[1]] - vals[names[0]]) > 0
        lines.append(f"| sessions where the LSNN is better ({label}) | | {int(better.sum())} of {len(sessions)} |")
    for key, label in (("parameters", "parameters"), ("cpu_ms_per_step", "CPU ms per 4 ms step")):
        lines.append(f"| {label} | " + " | ".join(f"{np.mean([results[s][m][key] for s in sessions]):,.3g}"
                                                 for m in names) + " |")
    with open(os.path.join(args.out_dir, "loso_comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Palatino", "TeX Gyre Pagella", "serif"], "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (key, label, _, _) in zip(axes, METRICS[:2]):
        a = np.array([results[s][names[0]][key] for s in sessions])
        b = np.array([results[s][names[1]][key] for s in sessions])
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.7", lw=1)
        ax.scatter(a, b, s=40, color="#C0392B", edgecolor="0.1", lw=0.6, zorder=3)
        ax.set_xlabel(f"SNN (LOSO): {label}")
        ax.set_ylabel(f"LSNN (LOSO): {label}")
        ax.set_title(label + (", below the line: LSNN better" if key == "rmse" else ", above the line: LSNN better"),
                     fontsize=12, loc="left")
    fig.suptitle(f"{args.experiment}/{args.subject}: one point per session ({len(sessions)})", x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loso_comparison.png"), dpi=200, facecolor="white")
    print(f"wrote {args.out_dir}/loso_comparison.json, .md, .png")


if __name__ == "__main__":
    main()
