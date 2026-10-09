#!/usr/bin/env python3
"""
Collect the best loss/RMSE from every sweep checkpoint, rank every
configuration, and group the configurations by architecture:

    depth (weight layers)  ->  width (peak hidden width)  ->  threshold  ->  decay

so e.g. all 8-layer configs can be compared directly against shallower ones.

Usage:
    python plot_snn_sweep_heatmap.py

Optional:
    python plot_snn_sweep_heatmap.py --checkpoint-root /path/to/no_tau_syn_sweep
    python plot_snn_sweep_heatmap.py --sweep-script run_snn_no_tau_syn_sweep.sbatch
    python plot_snn_sweep_heatmap.py --sort loss          # old loss-ranked row order
    python plot_snn_sweep_heatmap.py --common-sessions    # only sessions every config ran
    python plot_snn_sweep_heatmap.py --speck-compatible-only  # only configs with every
                                                              # hidden layer <= 256 wide

Outputs:
    snn_sweep_losses.csv              one row per (session, config)
    snn_sweep_config_summary.csv      one row per config, with architecture
                                      columns and overall / within-depth ranks
    snn_sweep_depth_summary.csv       one row per depth
    snn_sweep_depth_width_summary.csv one row per (depth, width)
    snn_sweep_best_by_depth.csv       per session: best loss reached by each depth
    <output>.png                      raw heatmap (rows grouped by architecture)
    <output>_normalized.png           per-session normalized heatmap
    <output>_by_depth.png             per-depth bar panels of config mean RMSE
                                      (colour = discrete peak width on RdBu_r,
                                      hatching = decay: 1 none, 2 ///, 3 xxx)
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/"
    "bmi/indy/no_tau_syn_sweep"
)
DEFAULT_SWEEP_SCRIPT = Path(__file__).resolve().parent / "run_snn_no_tau_syn_sweep.sbatch"

# Values the sbatch sweep uses for every config unless the row overrides them.
BASELINE_THRESHOLD = 1.0
BASELINE_DECAY = 2

# Largest hidden layer width the Speck2F chip can hold (--speck-compatible-only).
SPECK_MAX_WIDTH = 256

# Loss keys commonly stored in training checkpoints.
LOSS_KEYS = (
    "best_val_loss",
    "best_validation_loss",
    "val_loss",
    "validation_loss",
    "test_loss",
    "loss",
    "rmse",
    "best_rmse",
    "best_val_rmse",
    "val_rmse",
)


def scalar(value):
    """Convert a torch/numpy scalar to a Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.item())
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def find_losses(obj, prefix=""):
    """Recursively find scalar values whose key looks like a loss/RMSE."""
    found = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key).lower()
            full_key = f"{prefix}.{key}" if prefix else str(key)

            if any(k in key_str for k in LOSS_KEYS):
                val = scalar(value)
                if val is not None:
                    found[full_key] = val

            if isinstance(value, (dict, list, tuple)):
                found.update(find_losses(value, full_key))

    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list, tuple)):
                found.update(find_losses(value, f"{prefix}[{i}]"))

    return found


def select_best_loss(losses):
    """
    Pick the most meaningful loss among those found, by key name.
    Specific validation/test metrics come first; the generic "loss"
    substring match is the last resort (it would otherwise also match
    train_loss etc). Within a group the minimum value wins.
    """
    priority_groups = [
        ("val_rmse",),
        ("best_val_rmse",),
        ("best_val_loss",),
        ("best_validation_loss",),
        ("val_loss",),
        ("validation_loss",),
        ("test_loss",),
        ("rmse",),
        ("best_loss",),
        ("loss",),  # generic catch-all -- last resort only
    ]

    for group in priority_groups:
        candidates = {
            key: value
            for key, value in losses.items()
            if any(token in key.lower() for token in group)
        }
        if candidates:
            key = min(candidates, key=candidates.get)
            return candidates[key], key

    return None, None


# ---------------------------------------------------------------------------
# Configuration metadata (depth / width / threshold / decay)
# ---------------------------------------------------------------------------

# Matches SWEEP_TABLE rows:  "NAME|HIDDEN_DIMS|THRESHOLDS|DECAY"
_SWEEP_ROW = re.compile(
    r'^\s*"(?P<name>[^"|\s]+)\|(?P<dims>[\d ]+)\|(?P<thr>[\d. ]+)\|(?P<decay>\d+)"\s*$'
)


def make_meta(hidden_dims, thresholds, decay, source):
    """Build the architecture description used for grouping one config."""
    hidden_dims = [int(d) for d in hidden_dims]
    n_layers = len(hidden_dims) + 1  # weight layers = hidden + output
    thresholds = [float(t) for t in (thresholds or [BASELINE_THRESHOLD])]
    if len(thresholds) == 1:
        thresholds = thresholds * n_layers

    uniform = len(set(thresholds)) == 1
    if uniform:
        thr_label = f"{thresholds[0]:g}"
    else:
        thr_label = f"taper {thresholds[0]:g}->{thresholds[-1]:g}"

    decay = int(decay) if decay is not None else BASELINE_DECAY
    is_thr_variant = not (uniform and thresholds[0] == BASELINE_THRESHOLD)
    is_decay_variant = decay != BASELINE_DECAY
    if is_thr_variant and is_decay_variant:
        variant = "threshold+decay"
    elif is_thr_variant:
        variant = "threshold"
    elif is_decay_variant:
        variant = "decay"
    else:
        variant = "baseline"

    return {
        "depth": n_layers,
        "n_hidden": len(hidden_dims),
        "peak_width": max(hidden_dims),
        "total_hidden": sum(hidden_dims),
        "hidden_dims": " ".join(map(str, hidden_dims)),
        "threshold": thr_label,
        # Uniform thresholds sort numerically; tapered ones go after them.
        "threshold_sort": (0.0 if uniform else 1.0) * 100 + float(np.mean(thresholds)),
        "decay": decay,
        "variant": variant,
        "meta_source": source,
    }


def load_sweep_table(path):
    """Parse SWEEP_TABLE out of the sbatch script -> {config_name: meta}."""
    table = {}
    if path is None or not Path(path).exists():
        return table
    for line in Path(path).read_text().splitlines():
        m = _SWEEP_ROW.match(line)
        if m:
            table[m["name"]] = make_meta(
                m["dims"].split(), m["thr"].split(), m["decay"], "sweep_table"
            )
    return table


def meta_from_checkpoint(checkpoint):
    """Fallback: rebuild config metadata from the training args in a checkpoint."""
    if not isinstance(checkpoint, dict) or checkpoint.get("args") is None:
        return None
    a = checkpoint["args"]
    if not isinstance(a, dict):
        try:
            a = vars(a)
        except TypeError:
            return None
    dims = a.get("hidden_dims")
    if not dims:
        return None
    return make_meta(
        dims, a.get("spike_thresholds"), a.get("temporal_decay_stages"), "checkpoint_args"
    )


def unknown_meta(name):
    """Last resort when a config is neither in the table nor has usable args."""
    m = re.match(r"deep(\d+)_", name)
    return {
        "depth": float(m.group(1)) if m else np.nan,
        "n_hidden": np.nan,
        "peak_width": np.nan,
        "total_hidden": np.nan,
        "hidden_dims": "?",
        "threshold": "?",
        "threshold_sort": np.inf,
        "decay": np.nan,
        "variant": "unknown",
        "meta_source": "name_only",
    }


GROUP_SORT = ["depth", "peak_width", "threshold_sort", "decay", "mean_loss"]


def short_arch(r):
    """Compact architecture tag for plot labels, e.g. '8L w256 thr1 d2'."""
    def fmt(v):
        return "?" if pd.isna(v) else f"{int(v)}"
    return f"{fmt(r['depth'])}L w{fmt(r['peak_width'])} thr{r['threshold']} d{fmt(r['decay'])}"


def group_table(summary, keys):
    """Aggregate config-level results into one row per group."""
    rows = []
    for key, g in summary.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        best = g.loc[g["mean_loss"].idxmin()]
        row = dict(zip(keys, key))
        row.update(
            n_configs=len(g),
            best_config=best["config"],
            best_mean_loss=best["mean_loss"],
            best_overall_rank=int(best["rank_mean_loss"]),
            median_mean_loss=g["mean_loss"].median(),
            worst_mean_loss=g["mean_loss"].max(),
            avg_mean_rank=g["mean_rank"].mean(),
            total_top1=int(g["n_top1"].sum()),
            configs=", ".join(g.sort_values("mean_loss")["config"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def draw_group_separators(ax, ordered, n_cols):
    """Thick lines between depth groups, thin between widths; depth labels on the right."""
    depths = ordered["depth"].to_numpy()
    widths = ordered["peak_width"].to_numpy()
    n = len(ordered)
    for i in range(1, n):
        same_depth = depths[i] == depths[i - 1] or (pd.isna(depths[i]) and pd.isna(depths[i - 1]))
        if not same_depth:
            ax.axhline(i - 0.5, color="black", linewidth=2.5)
        elif widths[i] != widths[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.0, linestyle="--")

    start = 0
    for i in range(1, n + 1):
        boundary = i == n or not (
            depths[i] == depths[i - 1] or (pd.isna(depths[i]) and pd.isna(depths[i - 1]))
        )
        if boundary:
            d = depths[start]
            label = "? layers" if pd.isna(d) else f"{int(d)}-layer"
            ax.text(n_cols - 0.5 + 0.15, (start + i - 1) / 2, label,
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    rotation=270, clip_on=False)
            start = i


# Hatch pattern per temporal-decay value in the by-depth bar plot.
# Decays not listed here fall back to DECAY_HATCH_FALLBACK.
DECAY_HATCHES = {1: "", 2: "///", 3: "xxx"}
DECAY_HATCH_FALLBACK = "..."


def decay_hatch(decay):
    if pd.isna(decay):
        return DECAY_HATCH_FALLBACK
    return DECAY_HATCHES.get(int(decay), DECAY_HATCH_FALLBACK)


def width_colors(widths, cmap_name="RdBu_r", lo=0.1, hi=0.9):
    """
    One discrete colour per distinct peak width, spread evenly (by rank)
    between `lo` and `hi` on the colormap: narrowest -> blue, widest -> red.
    The extreme 10% at each end is skipped because RdBu's endpoints are so
    dark that black hatching becomes hard to see on them.
    """
    cmap = plt.get_cmap(cmap_name)
    uniq = sorted(int(w) for w in pd.Series(widths).dropna().unique())
    if len(uniq) == 1:
        return {uniq[0]: cmap(hi)}
    return {w: cmap(lo + (hi - lo) * i / (len(uniq) - 1)) for i, w in enumerate(uniq)}


def draw_width_key(fig, anchor_ax, color_of, square_in=0.17, gap_in=0.05,
                   pad_in=0.18, title="Peak hidden\nwidth"):
    """
    Discrete colour key: a vertical stack of small squares (widest on top),
    each labelled with its width, placed just right of `anchor_ax`.
    Returns the key axes so other legends can be positioned under it.
    """
    from matplotlib.patches import Rectangle

    fig_w, fig_h = fig.get_size_inches()
    widths = sorted(color_of, reverse=True)
    n = len(widths)
    step = square_in + gap_in
    key_h_in = n * step - gap_in
    key_w_in = square_in

    pos = anchor_ax.get_position()
    x0 = pos.x1 + pad_in / fig_w
    y0 = pos.y1 - key_h_in / fig_h - 0.04  # leave room for the title
    key_ax = fig.add_axes([x0, y0, key_w_in / fig_w, key_h_in / fig_h])

    for i, w in enumerate(widths):
        y = key_h_in - (i * step + square_in)
        key_ax.add_patch(Rectangle((0, y), square_in, square_in,
                                   facecolor=color_of[w], edgecolor="black", linewidth=0.5))
        key_ax.text(square_in + 0.06, y + square_in / 2, str(w),
                    ha="left", va="center", fontsize=8)

    key_ax.set_xlim(0, key_w_in)
    key_ax.set_ylim(0, key_h_in)
    key_ax.axis("off")
    key_ax.text(0, key_h_in + 0.08, title, ha="left", va="bottom", fontsize=8.5)
    return key_ax


def plot_bars_by_depth(summary, out_path, speck_only=False):
    """
    Side-by-side bar panels, one per depth, sharing a y axis. Within a panel
    bars are ordered threshold -> peak width -> decay. Bar height = mean RMSE
    across sessions, colour = peak hidden width (discrete RdBu_r: narrow = blue,
    wide = red), hatching = decay (1 = none, 2 = ///, 3 = xxx).
    """
    from matplotlib.patches import Patch

    plot_df = (
        summary.dropna(subset=["depth"])
        .sort_values(["depth", "threshold_sort", "peak_width", "decay", "mean_loss"])
        .reset_index(drop=True)
    )
    depths = sorted(plot_df["depth"].unique())
    counts = [int((plot_df["depth"] == d).sum()) for d in depths]

    color_of = width_colors(plot_df["peak_width"], "RdBu_r")

    fig, axes = plt.subplots(
        1, len(depths), sharey=True, squeeze=False,
        figsize=(max(10, 0.42 * len(plot_df) + 3), 5.5),
        gridspec_kw={"width_ratios": counts, "wspace": 0.06},
    )
    axes = axes[0]

    # Bars don't start at zero: the spread between configs is small relative
    # to the absolute RMSE, and would be invisible on a zero-based axis.
    lo, hi = plot_df["mean_loss"].min(), plot_df["mean_loss"].max()
    span = (hi - lo) or abs(hi) or 1.0
    y_bottom = max(0.0, lo - 0.15 * span)
    y_top = hi + 0.05 * span

    for ax, d in zip(axes, depths):
        g = plot_df[plot_df["depth"] == d].reset_index(drop=True)
        x = np.arange(len(g))
        colors = [color_of[int(w)] if not pd.isna(w) else "lightgrey"
                  for w in g["peak_width"]]
        bars = ax.bar(x, g["mean_loss"] - y_bottom, bottom=y_bottom, width=0.8,
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, decay in zip(bars, g["decay"]):
            bar.set_hatch(decay_hatch(decay))

        # Group ticks by threshold, with dotted separators between groups.
        thr = g["threshold"].tolist()
        starts = [0] + [i for i in range(1, len(g)) if thr[i] != thr[i - 1]]
        ends = starts[1:] + [len(g)]
        for s in starts[1:]:
            ax.axvline(s - 0.5, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
        ax.set_xticklabels([thr[s].replace("taper ", "taper\n").replace("->", "\u2192")
                            for s in starts], fontsize=7.5)
        ax.tick_params(axis="x", length=0)

        ax.set_xlim(-0.6, len(g) - 0.4)
        ax.set_title(f"{int(d)}-layer\n(n={len(g)})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylim(y_bottom, y_top)
    axes[0].set_ylabel("Mean RMSE across sessions  (lower is better)")
    fig.supxlabel("Spike threshold (within each depth: sorted by threshold, then width, then decay)",
                  fontsize=10)

    # Keys to the right of the last panel: discrete width squares on top,
    # decay hatching legend underneath.
    key_ax = None
    if color_of:
        key_ax = draw_width_key(fig, axes[-1], color_of)

    present = plot_df["decay"].dropna().astype(int).unique()
    handles = [Patch(facecolor="white", edgecolor="black", linewidth=0.5,
                     hatch=DECAY_HATCHES[k], label=f"{k}")
               for k in sorted(DECAY_HATCHES) if k in present]
    if any(k not in DECAY_HATCHES for k in present) or plot_df["decay"].isna().any():
        handles.append(Patch(facecolor="white", edgecolor="black", linewidth=0.5,
                             hatch=DECAY_HATCH_FALLBACK, label="other"))
    if handles:
        if key_ax is not None:
            kpos = key_ax.get_position()
            anchor = (kpos.x0 - 0.004, kpos.y0 - 0.06)
        else:
            apos = axes[-1].get_position()
            anchor = (apos.x1 + 0.01, apos.y1)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=anchor,
                   fontsize=8, frameon=False, handlelength=1.6, handleheight=1.6,
                   borderaxespad=0, title="Decay", title_fontsize=8.5,
                   alignment="left")

    fig.suptitle("Config RMSE by depth"
                 + (f" -- Speck-compatible only (width <= {SPECK_MAX_WIDTH})" if speck_only else ""))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing session/configuration checkpoint directories.",
    )
    parser.add_argument(
        "--sweep-script",
        type=Path,
        default=DEFAULT_SWEEP_SCRIPT,
        help="sbatch script whose SWEEP_TABLE defines each config's architecture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("snn_sweep_heatmap.png"),
        help="Output heatmap filename.",
    )
    parser.add_argument(
        "--sort",
        choices=["group", "loss"],
        default="group",
        help="Heatmap row order: grouped by depth/width/threshold/decay (default) "
             "or purely by mean loss.",
    )
    parser.add_argument(
        "--common-sessions",
        action="store_true",
        help="Only use sessions in which EVERY config has a result, so all "
             "configs are compared on exactly the same sessions.",
    )
    parser.add_argument(
        "--speck-compatible-only",
        action="store_true",
        help=f"Only include configs whose widest hidden layer is <= {SPECK_MAX_WIDTH} "
             f"neurons (Speck2F-deployable). Applied before ranking, so all ranks are "
             f"among compatible configs only.",
    )
    args = parser.parse_args()

    root = args.checkpoint_root.expanduser()

    # Filtered runs write to separately-named files so they don't overwrite
    # the full-sweep results.
    tag = "_speck" if args.speck_compatible_only else ""
    if tag:
        args.output = args.output.with_name(args.output.stem + tag + args.output.suffix)

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist:\n{root}")

    sweep_table = load_sweep_table(args.sweep_script.expanduser())
    if sweep_table:
        print(f"Loaded {len(sweep_table)} config definitions from {args.sweep_script}")
    else:
        print(f"[warning] no SWEEP_TABLE found at {args.sweep_script}; "
              f"falling back to the 'args' stored in each checkpoint.")

    # The sbatch sweep has the structure:
    # root / session_id / sweep_name / *.pth
    checkpoint_files = sorted(root.glob("*/*/*.pth"))

    if not checkpoint_files:
        raise RuntimeError(f"No .pth checkpoints found under:\n{root}")

    rows = []
    args_meta = {}  # config -> meta rebuilt from checkpoint args (fallback only)

    for path in checkpoint_files:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            print(f"[warning] could not read {path}: {exc}")
            continue

        # Use the explicit top-level best_loss when present; otherwise search
        # the checkpoint for the most meaningful loss key instead of crashing
        # with a KeyError (e.g. on a weights-only file).
        best_loss, loss_key = None, None
        if isinstance(checkpoint, dict):
            best_loss = scalar(checkpoint.get("best_loss"))
            loss_key = "best_loss" if best_loss is not None else None
        if best_loss is None:
            best_loss, loss_key = select_best_loss(find_losses(checkpoint))

        if best_loss is None:
            print(f"[warning] no loss/RMSE found in {path}")
            continue

        # root/session/config/model.pth
        session = path.parent.parent.name
        config = path.parent.name

        if config not in sweep_table and config not in args_meta:
            meta = meta_from_checkpoint(checkpoint)
            if meta is not None:
                args_meta[config] = meta

        rows.append(
            {
                "session": session,
                "config": config,
                "loss": best_loss,
                "loss_key": loss_key,
                "checkpoint": str(path),
            }
        )

    if not rows:
        raise RuntimeError(
            "No checkpoints contained a recognizable loss/RMSE value. "
            "Inspect one .pth file to determine its metadata keys."
        )

    df = pd.DataFrame(rows)

    # If multiple checkpoints exist for one (session, config) pair, keep the
    # lowest loss.
    df = (
        df.sort_values("loss")
        .drop_duplicates(subset=["session", "config"], keep="first")
        .reset_index(drop=True)
    )

    def config_meta(c):
        meta = sweep_table.get(c) or args_meta.get(c)
        if meta is None:
            print(f"[warning] no architecture info for config '{c}'; grouped as unknown.")
            meta = unknown_meta(c)
        return meta

    meta_by_config = {c: config_meta(c) for c in sorted(df["config"].unique())}

    if args.speck_compatible_only:
        # Unknown widths (NaN) cannot be verified as compatible, so they are
        # excluded too rather than silently assumed to fit.
        keep = {c for c, m in meta_by_config.items()
                if not pd.isna(m["peak_width"]) and m["peak_width"] <= SPECK_MAX_WIDTH}
        excluded = sorted(set(meta_by_config) - keep)
        if not keep:
            raise RuntimeError(f"--speck-compatible-only: no config has every hidden "
                               f"layer <= {SPECK_MAX_WIDTH} neurons.")
        df = df[df["config"].isin(keep)].reset_index(drop=True)
        print(f"--speck-compatible-only: kept {len(keep)} config(s) with max hidden width "
              f"<= {SPECK_MAX_WIDTH}; excluded {len(excluded)}:")
        for c in excluded:
            w = meta_by_config[c]["peak_width"]
            print(f"    {c:<22} peak width {'unknown' if pd.isna(w) else int(w)}")

    # ---------------------------------------------------------------------
    # Session coverage. Deeper configs were added to the table later, so they
    # may have run on fewer sessions than the original ones; mean losses over
    # different session sets are not directly comparable. Warn, and optionally
    # restrict to sessions every config has.
    # ---------------------------------------------------------------------
    all_configs = sorted(df["config"].unique())
    coverage = df.groupby("session")["config"].nunique()
    full_sessions = coverage[coverage == len(all_configs)].index.tolist()
    if args.common_sessions:
        if not full_sessions:
            raise RuntimeError(
                "--common-sessions: no session has results for every config. "
                "Run without it, or filter configs first."
            )
        dropped = sorted(set(df["session"]) - set(full_sessions))
        df = df[df["session"].isin(full_sessions)].reset_index(drop=True)
        print(f"--common-sessions: using {len(full_sessions)} session(s) with all "
              f"{len(all_configs)} configs; dropped {len(dropped)}: {dropped}")
    else:
        per_config = df.groupby("config")["session"].nunique()
        if per_config.nunique() > 1:
            print(f"[note] configs ran on different numbers of sessions "
                  f"({per_config.min()}-{per_config.max()}). Mean loss compares "
                  f"different session sets; mean_rank is more robust to this. "
                  f"Use --common-sessions for a strictly matched comparison "
                  f"({len(full_sessions)} session(s) currently have every config).")

    df.to_csv(f"snn_sweep_losses{tag}.csv", index=False)

    # ---------------------------------------------------------------------
    # Per-config summary: mean across sessions, plus within-session ranks.
    # ---------------------------------------------------------------------
    summary = (
        df.groupby("config")["loss"]
        .agg(mean_loss="mean", std_loss="std", min_loss="min", max_loss="max", n_sessions="count")
        .reset_index()
        .sort_values("mean_loss")
        .reset_index(drop=True)
    )
    summary["std_loss"] = summary["std_loss"].fillna(0.0)  # a single-session config has no std

    sessions_sorted = sorted(df["session"].unique())

    # Full (config x session) matrix, NaN where a session has no checkpoint
    # for that config -- rendered as blank, not zero.
    pivot = df.pivot_table(index="config", columns="session", values="loss", aggfunc="min")

    # Rank configs WITHIN each session (1 = lowest loss), independent of that
    # session's absolute loss scale; NaN cells are skipped.
    ranks = pivot.rank(axis=0, method="min")
    n_top1 = (ranks == 1).sum(axis=1)
    mean_rank = ranks.mean(axis=1, skipna=True)
    n_ranked = ranks.notna().sum(axis=1)

    summary["n_top1"] = summary["config"].map(n_top1)
    summary["top1_rate"] = summary["n_top1"] / summary["config"].map(n_ranked)
    summary["mean_rank"] = summary["config"].map(mean_rank)

    # Overall ranks (1 = best) by both criteria.
    summary["rank_mean_loss"] = summary["mean_loss"].rank(method="min").astype(int)
    summary["rank_mean_rank"] = summary["mean_rank"].rank(method="min").astype(int)

    # Attach architecture metadata.
    summary = pd.concat(
        [summary, pd.DataFrame([meta_by_config[c] for c in summary["config"]])], axis=1
    )

    # Rank within each depth group (1 = best of that depth).
    summary["rank_in_depth"] = (
        summary.groupby("depth", dropna=False)["mean_loss"].rank(method="min").astype(int)
    )

    # Final column order: identity -> architecture -> results.
    col_order = [
        "config", "depth", "peak_width", "threshold", "decay", "variant",
        "hidden_dims", "n_hidden", "total_hidden",
        "rank_mean_loss", "rank_in_depth", "rank_mean_rank",
        "mean_loss", "std_loss", "min_loss", "max_loss", "n_sessions",
        "mean_rank", "n_top1", "top1_rate", "meta_source", "threshold_sort",
    ]
    summary = summary[col_order]
    summary_grouped = summary.sort_values(GROUP_SORT, na_position="last").reset_index(drop=True)
    summary_grouped.drop(columns="threshold_sort").to_csv(f"snn_sweep_config_summary{tag}.csv", index=False)

    # Group-level tables.
    depth_summary = group_table(summary, ["depth"])
    depth_width_summary = group_table(summary, ["depth", "peak_width"])
    depth_summary.to_csv(f"snn_sweep_depth_summary{tag}.csv", index=False)
    depth_width_summary.to_csv(f"snn_sweep_depth_width_summary{tag}.csv", index=False)

    # Per session: the best loss any config of each depth reached, and how
    # the depths rank against each other in that session.
    depth_of = summary.set_index("config")["depth"]
    best_by_depth = (
        df.assign(depth=df["config"].map(depth_of))
        .groupby(["session", "depth"])["loss"].min()
        .unstack("depth")
        .reindex(sessions_sorted)
    )
    depth_ranks = best_by_depth.rank(axis=1, method="min")
    best_by_depth.to_csv(f"snn_sweep_best_by_depth{tag}.csv")

    # ---------------------------------------------------------------------
    # Heatmaps
    # ---------------------------------------------------------------------
    if args.sort == "group":
        ordered = summary_grouped
    else:
        ordered = summary.sort_values("mean_loss").reset_index(drop=True)
    configs_sorted = ordered["config"].tolist()
    pivot = pivot.reindex(index=configs_sorted, columns=sessions_sorted)

    row_labels = [
        f"#{int(r['rank_mean_loss']):>2}  {r['config']}   [{short_arch(r)}]"
        for _, r in ordered.iterrows()
    ]

    # Per-session min-max normalization for the second heatmap
    # (0 = best config in that session, 1 = worst).
    col_min = pivot.min(axis=0, skipna=True)
    col_max = pivot.max(axis=0, skipna=True)
    col_range = (col_max - col_min).replace(0, np.nan)
    pivot_norm = (pivot - col_min) / col_range

    fig_height = max(5, 0.3 * len(configs_sorted) + 2)
    fig_width = max(10, 1.1 * (len(sessions_sorted) + 1) + 6)

    mean_col = ordered.set_index("config").loc[configs_sorted, "mean_loss"]
    matrix = np.column_stack([pivot.to_numpy(), mean_col.to_numpy()])
    col_labels = sessions_sorted + ["MEAN"]

    order_desc = ("grouped by depth -> width -> threshold -> decay"
                  if args.sort == "group" else "sorted by mean loss")
    if args.speck_compatible_only:
        order_desc += f", Speck-compatible only (width <= {SPECK_MAX_WIDTH})"

    def draw_heatmap(ax, color_matrix, **imshow_kwargs):
        im = ax.imshow(np.ma.masked_invalid(color_matrix), aspect="auto", **imshow_kwargs)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(configs_sorted)))
        ax.set_yticklabels(row_labels, fontsize=8, family="monospace")
        ax.axvline(len(sessions_sorted) - 0.5, color="white", linewidth=2)
        # Text always shows RAW loss values; only the colour scale differs.
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isnan(value):
                    continue
                weight = "bold" if col == matrix.shape[1] - 1 else "normal"
                ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                        fontsize=7, fontweight=weight)
        if args.sort == "group":
            draw_group_separators(ax, ordered, len(col_labels))
        return im

    best = summary.sort_values("mean_loss").iloc[0]
    best_config_rows = df[df["config"] == best["config"]]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = draw_heatmap(ax, matrix)
    ax.set_title(
        f"SNN sweep results -- {len(sessions_sorted)} session(s) x {len(configs_sorted)} config(s), "
        f"{order_desc}\n"
        f"#N = overall rank by mean loss.  Best: {best['config']} "
        f"(mean={best['mean_loss']:.5f}, n={int(best['n_sessions'])})"
    )
    ax.set_xlabel("Session (MEAN = average across all sessions with this config) -- lower is better")
    fig.colorbar(im, ax=ax, label="Loss / RMSE", pad=0.06)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.show()

    mean_norm = (mean_col - mean_col.min()) / (mean_col.max() - mean_col.min()) \
        if mean_col.max() > mean_col.min() else mean_col * 0
    matrix_norm = np.column_stack([pivot_norm.to_numpy(), mean_norm.to_numpy()])

    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    im2 = draw_heatmap(ax2, matrix_norm, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax2.set_title(
        f"SNN sweep results, PER-SESSION NORMALIZED -- {order_desc}\n"
        f"Color = position within each session's own range (green=best, red=worst); "
        f"numbers = raw loss"
    )
    ax2.set_xlabel("Session (MEAN column normalized across configs' own mean_loss)")
    fig2.colorbar(im2, ax=ax2, label="Normalized within session (0=best, 1=worst)", pad=0.06)
    fig2.tight_layout()
    normalized_output = args.output.with_name(args.output.stem + "_normalized" + args.output.suffix)
    fig2.savefig(normalized_output, dpi=200, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------
    # Depth comparison figure: one bar panel per depth, bars sorted by
    # threshold, then peak width, then decay.
    # ---------------------------------------------------------------------
    depth_output = args.output.with_name(args.output.stem + "_by_depth" + args.output.suffix)
    plot_bars_by_depth(summary, depth_output, args.speck_compatible_only)
    plt.show()

    # ---------------------------------------------------------------------
    # Text report
    # ---------------------------------------------------------------------
    n_total = len(summary)
    print("\n============================================================")
    print("ALL CONFIGS, GROUPED: depth -> width -> threshold -> decay")
    print(f"(#N = overall rank by mean loss out of {n_total}; "
          f"d#N = rank within its depth group)")
    print("============================================================")
    for depth, g_depth in summary_grouped.groupby("depth", sort=False, dropna=False):
        d_label = "unknown depth" if pd.isna(depth) else f"{int(depth)}-LAYER"
        print(f"\n{d_label}  ({len(g_depth)} configs, best overall rank "
              f"#{int(g_depth['rank_mean_loss'].min())}, "
              f"median mean loss {g_depth['mean_loss'].median():.5f})")
        for width, g_w in g_depth.groupby("peak_width", sort=False, dropna=False):
            w_label = "?" if pd.isna(width) else int(width)
            print(f"  peak width {w_label}:")
            for _, r in g_w.iterrows():
                print(f"    #{int(r['rank_mean_loss']):>2}  d#{int(r['rank_in_depth']):<2} "
                      f"{r['config']:<22} thr={r['threshold']:<15} decay={r['decay']!s:<3} "
                      f"mean={r['mean_loss']:.5f}  std={r['std_loss']:.5f}  "
                      f"n={int(r['n_sessions'])}  mean_rank={r['mean_rank']:.2f}  "
                      f"top1={int(r['n_top1'])}")

    print("\n============================================================")
    print("DEPTH SUMMARY")
    print("============================================================")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(depth_summary.drop(columns="configs").to_string(index=False, float_format="%.5f"))

    print("\nBest config of each depth, per session (which depth wins each session):")
    wins = (depth_ranks == 1).sum(axis=0)
    for d in best_by_depth.columns:
        print(f"  {int(d)}-layer: best-of-depth wins {int(wins[d])}/"
              f"{int(best_by_depth[d].notna().sum())} sessions, "
              f"mean depth rank {depth_ranks[d].mean():.2f}")
    print("  (Note: depths with more configs get more 'draws' at the per-session best.)")

    print("\n============================================================")
    print("BEST CONFIGURATION (ranked by mean loss across sessions)")
    print("============================================================")
    print(f"Config:        {best['config']}  [{short_arch(best)}]")
    print(f"Mean loss:     {best['mean_loss']:.6f}  (std={best['std_loss']:.6f}, "
          f"n={int(best['n_sessions'])} session(s))")
    print(f"Min/Max loss:  {best['min_loss']:.6f} / {best['max_loss']:.6f}")
    print(f"Top-1 finishes: {int(best['n_top1'])}/{int(n_ranked[best['config']])} sessions "
          f"({best['top1_rate']*100:.0f}%), mean rank {best['mean_rank']:.2f}")
    print("Per-session breakdown for this config:")
    for _, r in best_config_rows.sort_values("session").iterrows():
        print(f"    {r['session']:<28} loss={r['loss']:.6f}  (source: {r['loss_key']})")

    most_frequent_winner = summary.sort_values(["n_top1", "top1_rate"], ascending=False).iloc[0]
    print(f"\nMost frequent top-1 finisher across sessions: {most_frequent_winner['config']} "
          f"({int(most_frequent_winner['n_top1'])}/{int(n_ranked[most_frequent_winner['config']])} sessions, "
          f"{most_frequent_winner['top1_rate']*100:.0f}%)")
    if most_frequent_winner["config"] != best["config"]:
        print(f"  NOTE: this differs from the best-mean-loss config ({best['config']}) -- "
              f"worth looking at both rather than treating either alone as decisive.")

    print(f"\nSaved per-(session,config) results:  {Path(f'snn_sweep_losses{tag}.csv').resolve()}")
    print(f"Saved per-config summary (grouped):  {Path(f'snn_sweep_config_summary{tag}.csv').resolve()}")
    print(f"Saved depth summary:                 {Path(f'snn_sweep_depth_summary{tag}.csv').resolve()}")
    print(f"Saved depth x width summary:         {Path(f'snn_sweep_depth_width_summary{tag}.csv').resolve()}")
    print(f"Saved best-of-depth per session:     {Path(f'snn_sweep_best_by_depth{tag}.csv').resolve()}")
    print(f"Saved plot (raw values):             {args.output.resolve()}")
    print(f"Saved plot (per-session normalized): {normalized_output.resolve()}")
    print(f"Saved plot (by depth):               {depth_output.resolve()}")


if __name__ == "__main__":
    main()
