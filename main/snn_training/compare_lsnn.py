"""
Train the existing SNN and the recurrent LSNN (train_snn.py --arch lsnn) on one
session and compare them side by side on its held-out test split.

Both runs use train_snn.py with the same data, epochs, early stopping, optimizer,
learning rate, batch size and binarized input; only the network differs:

    snn   the per-session SNN of the final pipeline (sbatch_scripts/snn_config.sh,
          select_model per_session): feedforward 256 -> 128 -> 64 IAF, hard reset,
          thresholds 1.0, 2 EMA readout stages, no tau_syn.
          --snn-config pooled instead trains the pooled architecture (512 -> 256 ->
          128, tau_syn 2 4 8 16, hard reset) on this session alone; pooled_soft
          the same with soft reset, the LOSO fine-tuned model's exact settings
          without the pretraining.
    lsnn  2 x 256 recurrent IAF (--lsnn-hidden-dims), 90% with threshold adaptation
          (500 ms), per-neuron synaptic filters 20 ms-1.3 s, soft reset, up to 3
          spikes per step, leaky linear readout (models/model_bmi_lsnn.py).

Outputs in --out-dir:
    snn_per_session/ (or snn_pooled/), lsnn/   train_snn.py's checkpoints, curves and summaries
    comparison_{snn-config}.json, .md, .png    test metrics and velocity (first --plot-seconds)
Both --snn-config values can share one --out-dir: the LSNN is then trained once.

Example (one indy session, CPU, under an hour per configuration):
    python snn_training/compare_lsnn.py \\
        --data-path $BMI_DATA_ROOT/snn_datasets/bmi/indy/mua_8_group/indy_20160411_01 \\
        --out-dir lsnn_vs_snn/indy_20160411_01 --device cpu [--snn-config pooled]
Re-running skips a model whose best_model_weights.pth exists (--retrain to redo).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from dataset import CustomDataset  # noqa: E402
from train_snn import STEP_MS, build_model, unscale_velocity  # noqa: E402

POOLED = ["--hidden-dims", "512", "256", "128", "--spike-thresholds", "1.0", "1.0", "1.0", "1.0",
          "--tau-syn", "2", "4", "8", "16"]
SNN_CONFIGS = {   # (reset, network flags)
    "per_session": ("hard", ["--hidden-dims", "256", "128", "64", "--spike-thresholds", "1.0", "1.0", "1.0", "1.0"]),
    "pooled": ("hard", POOLED),          # the pooled network as pretrained (hard reset)
    "pooled_soft": ("soft", POOLED),     # the pooled network as fine-tuned (soft reset): the LOSO model's settings
    "pooled_soft_unitgain": ("soft", POOLED + ["--unit-gain-syn", "true"]),   # the same, unit-gain synaptic filters
}


def train(name, extra, args):
    """Run train_snn.py for one model unless its best checkpoint exists; returns its directory."""
    ckpt_dir = os.path.join(args.out_dir, name)
    best = os.path.join(ckpt_dir, "best_model_weights.pth")
    if os.path.isfile(best) and not args.retrain:
        print(f"[{name}] {best} exists, skipping training")
        return ckpt_dir
    cmd = [sys.executable, os.path.join(HERE, "train_snn.py"),
           "--config", os.path.join(os.path.dirname(HERE), "configs", "iaf_hard_reset.yaml"),
           "--data-path", args.data_path, "--checkpoint-dir", ckpt_dir, "--device", args.device,
           "--epochs", str(args.epochs), "--patience", str(args.patience),
           "--scheduler-patience", str(args.scheduler_patience), "--checkpoint-interval", "10",
           "--batch-size", "20", "--num-workers", "0", "--binarize-input", "true",
           "--weight-init", "kaiming", "--seed", str(args.seed)]
    if args.train_data_min:
        cmd += ["--train-data-min", str(args.train_data_min)]
    cmd += extra
    print(f"[{name}] " + " ".join(cmd), flush=True)
    t0 = time.time()
    # Agg: utils/visualization.py ends with plt.show(), which blocks on a desktop backend
    subprocess.run(cmd, check=True, env={**os.environ, "MPLBACKEND": "Agg"})
    print(f"[{name}] trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    return ckpt_dir


def evaluate(ckpt_dir, device):
    """Best checkpoint on the test split: predictions, metrics, spike rates, compute time per step."""
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model_weights.pth"), map_location=device, weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    experiment = ckpt.get("experiment", "bmi")
    module, model = build_model(args, experiment, ckpt["input_shape"][0], device)
    if getattr(args, "arch", "snn") == "snn":
        model(torch.zeros(1, 1, ckpt["input_shape"][0], device=device))   # gives sinabs' state buffers their shape
    module.load_model_weights(model, ckpt["model_state_dict"], neuron_type=args.neuron_type,
                              source_description=ckpt_dir)
    model.eval()
    test = CustomDataset(args.data_path, train=False, experiment=experiment, binarize=args.binarize_input)
    scale = (args.velocity_lo, args.velocity_hi, args.velocity_margin)
    preds, trues, spikes, steps, seconds = [], [], None, 0, 0.0
    with torch.no_grad():
        for i in range(len(test)):
            _, x, v = test[i]
            x = x.unsqueeze(1).to(device)                                   # (T, 1, channels)
            t0 = time.time()
            out, _, _ = model(x)
            seconds += time.time() - t0
            steps += x.shape[0]
            rates = [float(s) / (x.shape[0] * w) for s, w in zip(model.layer_spikes, model.layer_widths)]
            spikes = rates if spikes is None else [a + b for a, b in zip(spikes, rates)]
            preds.append(unscale_velocity(out[:, 0].cpu().numpy(), *scale))
            trues.append(unscale_velocity(v.numpy(), *scale))
    pred, true = np.concatenate(preds), np.concatenate(trues)
    err = pred - true
    r = [float(np.corrcoef(pred[:, k], true[:, k])[0, 1]) for k in range(2)]
    r2 = [float(1 - (err[:, k] ** 2).sum() / ((true[:, k] - true[:, k].mean()) ** 2).sum()) for k in range(2)]
    return {
        "arch": getattr(args, "arch", "snn"),
        "hidden_dims": args.hidden_dims,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs_trained": int(ckpt["epoch"]) + 1,
        "rmse": float(np.sqrt((err ** 2).sum(axis=1).mean())),             # speed error, as train_snn.py's velocity_error
        "rmse_per_axis": [float(np.sqrt((err[:, k] ** 2).mean())) for k in range(2)],
        "r_xy": r, "r_mean": float(np.mean(r)), "r2_xy": r2, "r2_mean": float(np.mean(r2)),
        "spikes_per_neuron_per_step": [s / len(test) for s in spikes],
        "cpu_ms_per_step": 1000 * seconds / steps,
        "test_seconds": steps * STEP_MS / 1000,
    }, pred, true


def write_outputs(results, traces, args):
    tag = f"comparison_{args.snn_config}" + ("" if args.lsnn_name == "lsnn" else f"_{args.lsnn_name}")
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    rows = [("Network", lambda r: f"{r['arch']} {r['hidden_dims']}"), ("Parameters", lambda r: f"{r['parameters']:,}"),
            ("Epochs (best)", lambda r: str(r["epochs_trained"])),
            ("Velocity error (RMS, test)", lambda r: f"{r['rmse']:.1f}"),
            ("Correlation r (x, y)", lambda r: f"{r['r_xy'][0]:.3f}, {r['r_xy'][1]:.3f}"),
            ("R² (x, y)", lambda r: f"{r['r2_xy'][0]:.3f}, {r['r2_xy'][1]:.3f}"),
            ("Spikes per neuron per step", lambda r: ", ".join(f"{s:.3f}" for s in r["spikes_per_neuron_per_step"])),
            ("CPU time per 4 ms step", lambda r: f"{r['cpu_ms_per_step']:.3f} ms")]
    names = list(results)
    lines = [f"# SNN vs LSNN: {os.path.basename(os.path.normpath(args.data_path))}", "",
             f"Test split: {results[names[0]]['test_seconds']:.0f} s. Velocity in the dataset's physical units.", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {lab} | " + " | ".join(fn(results[n]) for n in names) + " |" for lab, fn in rows]
    with open(os.path.join(args.out_dir, f"{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Palatino", "TeX Gyre Pagella", "serif"], "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"snn": "0.55", "lsnn": "#C0392B"}
    n = int(args.plot_seconds * 1000 / STEP_MS)
    t = np.arange(n) * STEP_MS / 1000
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    true = traces["true"][:n]
    for k, (ax, lab) in enumerate(zip(axes, ("x velocity", "y velocity"))):
        ax.plot(t, true[:, k], color="0.1", lw=1.6, label="true")
        for name in names:
            r = results[name]
            ax.plot(t, traces[name][:n, k], color=colors[r["arch"]], lw=1.3,
                    label=f"{name} (r = {r['r_xy'][k]:.2f})")
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=3)
    axes[-1].set_xlabel("time in the test split (s)")
    fig.suptitle(f"{os.path.basename(os.path.normpath(args.data_path))}: test velocity, first {args.plot_seconds:g} s",
                 x=0.06, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{tag}.png"), dpi=200, facecolor="white")
    print(f"wrote {args.out_dir}/{tag}.json, .md, .png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="one session's SNN dataset (train/ and test/)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50, help="maximum epochs (early stopping), as per_session")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler-patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-data-min", type=float, default=None, help="train on the first N minutes only")
    p.add_argument("--snn-config", default="per_session", choices=list(SNN_CONFIGS))
    p.add_argument("--lsnn-hidden-dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lsnn-name", default="lsnn", help="the LSNN's checkpoint directory and table column")
    p.add_argument("--lsnn-args", nargs=argparse.REMAINDER, default=[],
                   help="more train_snn.py flags for the LSNN only (last on the line)")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--plot-seconds", type=float, default=10.0)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    reset, net = SNN_CONFIGS[args.snn_config]
    snn_flags = ["--arch", "snn", "--neuron-type", "iaf", "--reset-type", reset, "--spike-fn", "multi",
                 "--min-vmem", "-1", "--use-iaf-squeeze", "--temporal-decay-stages", "2"] + net
    lsnn_flags = ["--arch", "lsnn", "--hidden-dims", *map(str, args.lsnn_hidden_dims)] + args.lsnn_args
    results, traces = {}, {}
    for name, flags in ((f"snn_{args.snn_config}", snn_flags), (args.lsnn_name, lsnn_flags)):
        ckpt_dir = train(name, flags, args)
        results[name], traces[name], traces["true"] = evaluate(ckpt_dir, torch.device(args.device))
    write_outputs(results, traces, args)


if __name__ == "__main__":
    main()
