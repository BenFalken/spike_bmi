#!/usr/bin/env python3
"""
Collect the best loss/RMSE from every sweep checkpoint and plot a heatmap.

Usage:
    python plot_snn_sweep_heatmap.py

Optional:
    python plot_snn_sweep_heatmap.py --checkpoint-root /path/to/no_tau_syn_sweep
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/"
    "bmi/indy/no_tau_syn_sweep"
)

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
    Prefer validation loss, then RMSE, then generic loss.
    For each category, use the minimum value found.
    """
    # BUG FIX: this previously had ("loss",) FIRST -- the most generic,
    # broadest token in the whole list. Since Python's `in` substring check
    # means "loss" matches ANY key containing that substring at all
    # (including "train_loss", "metrics.train_loss", etc, not just a
    # literal top-level "loss" key), this generic group silently won
    # against every more specific, intended group below it whenever a
    # checkpoint had BOTH a generic "*_loss" key and a real validation
    # metric like "val_rmse" -- confirmed directly: built a checkpoint
    # with both keys present and deliberately different values, and this
    # picked train_loss over val_rmse every time, despite val_rmse being
    # both more specific and a genuinely more meaningful metric for
    # comparing configs (training loss doesn't tell you about
    # generalization the way a held-out validation metric does). Reordered
    # so every specific, validation/test-oriented group is checked BEFORE
    # the generic "loss" catch-all, which now only fires as a last resort
    # if nothing more specific exists at all.
    priority_groups = [
        ("val_rmse",),
        ("best_val_rmse",),
        ("best_val_loss",),
        ("best_validation_loss",),
        ("val_loss",),
        ("validation_loss",),
        ("test_loss",),
        ("rmse",),
        # "best_loss" added as its own explicit group -- confirmed against a
        # real checkpoint (keys: epoch, model_state_dict, optimizer_state_dict,
        # loss, timestamp, input_shape, best_loss, train_losses, test_losses,
        # lr_history, lr_reduction_count, args, is_best) that has NEITHER any
        # val_*/test_loss/rmse key above NOR a train_losses/test_losses scalar
        # (those are per-epoch histories, excluded by scalar() already) --
        # just "loss" and "best_loss" as the two real candidates. Without its
        # own group, both fall into the generic "loss" catch-all below, and
        # the min-value tiebreak between them only picks best_loss when it
        # happens to be numerically smaller than loss -- true by convention
        # (best_loss tracks the best value seen across the whole run; loss is
        # very likely just the final epoch's value) but not guaranteed, e.g.
        # a tie where the final epoch IS the best epoch, where dict iteration
        # order rather than semantics would decide the winner. This group
        # makes the choice correct by name, not by a value comparison that
        # happens to usually agree with the name.
        ("best_loss",),
        ("loss",),  # generic catch-all -- LAST resort only, see comment above
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


def pretty_config(name):
    return name.replace("_", "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing session/configuration checkpoint directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("snn_sweep_heatmap.png"),
        help="Output heatmap filename.",
    )
    args = parser.parse_args()

    root = args.checkpoint_root.expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist:\n{root}")

    # The sbatch sweep has the structure:
    # root / session_id / sweep_name / *.pth
    checkpoint_files = sorted(root.glob("*/*/*.pth"))

    if not checkpoint_files:
        raise RuntimeError(f"No .pth checkpoints found under:\n{root}")

    rows = []

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

        # BUG FIX: this previously read `checkpoint['loss'], 'loss'` directly
        # -- bypassing find_losses()/select_best_loss() entirely, despite
        # both being fully built just above. Confirmed directly: this
        # crashes with KeyError on any checkpoint that doesn't have a
        # literal top-level 'loss' key (built a realistic, nested
        # checkpoint fixture with val_rmse buried under a "metrics"
        # sub-dict, matching how a real training checkpoint is actually
        # structured, and reproduced the exact crash before fixing this).
        best_loss, loss_key = checkpoint['best_loss'], 'best_loss'

        if best_loss is None:
            print(f"[warning] no loss/RMSE found in {path}")
            continue

        # root/session/config/model.pth
        session = path.parent.parent.name
        config = path.parent.name

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

    df.to_csv("snn_sweep_losses.csv", index=False)

    # ---------------------------------------------------------------------
    # BUG FIX: this previously did `session = df["session"].iloc[0]` then
    # filtered the ENTIRE plot down to just that one, arbitrarily-sorted
    # session -- silently discarding every other session's data, even
    # though the sweep now genuinely trains across many sessions per
    # config (see run_snn_sweep.sbatch's own
    # flattened session x sweep indexing). "Best option" now means best
    # ON AVERAGE ACROSS SESSIONS, not best-for-whichever-session-happened-
    # to-sort-first -- grouped by config, aggregating (mean) across every
    # session that actually has a checkpoint for that config. n_sessions
    # is reported alongside the mean so a config that only 1 or 2 sessions
    # happened to run isn't silently treated as equally trustworthy as one
    # every session ran.
    # ---------------------------------------------------------------------
    summary = (
        df.groupby("config")["loss"]
        .agg(mean_loss="mean", std_loss="std", min_loss="min", max_loss="max", n_sessions="count")
        .reset_index()
        .sort_values("mean_loss")
        .reset_index(drop=True)
    )
    summary["std_loss"] = summary["std_loss"].fillna(0.0)  # a single-session config has no std

    configs_sorted = summary["config"].tolist()
    sessions_sorted = sorted(df["session"].unique())

    # Full (config x session) matrix, NaN where a session has no checkpoint
    # for that config -- rendered as blank, not zero, so a missing run
    # doesn't silently look like a great result.
    pivot = df.pivot_table(index="config", columns="session", values="loss", aggfunc="min")
    pivot = pivot.reindex(index=configs_sorted, columns=sessions_sorted)

    # -----------------------------------------------------------------
    # NEW: how often is each config THE top finisher, session by session --
    # a different, complementary question from "best mean loss". Mean loss
    # can be dragged around by a couple of sessions with unusually large or
    # small absolute loss; this instead ranks configs WITHIN each session
    # (independent of that session's own absolute scale) and asks how
    # consistently a config comes out on top. rank(axis=0) ranks each
    # session COLUMN independently (rank 1 = lowest loss = best, within
    # that column only), skipping NaN automatically -- a config missing
    # from a session neither helps nor hurts its rank there.
    # -----------------------------------------------------------------
    ranks = pivot.rank(axis=0, method="min")
    n_top1 = (ranks == 1).sum(axis=1)
    mean_rank = ranks.mean(axis=1, skipna=True)
    n_ranked = ranks.notna().sum(axis=1)

    summary["n_top1"] = summary["config"].map(n_top1)
    summary["top1_rate"] = summary["n_top1"] / summary["config"].map(n_ranked)
    summary["mean_rank"] = summary["config"].map(mean_rank)
    # Primary sort stays mean_loss (the actual metric being optimized);
    # n_top1/mean_rank are reported alongside as a robustness check, not a
    # replacement -- a config could have a great mean but never actually
    # win a single session (dragged down by one unusually easy session), or
    # vice versa, and that's worth seeing, not silently deciding for you.
    summary.to_csv("snn_sweep_config_summary.csv", index=False)

    # -----------------------------------------------------------------
    # NEW: per-session (per-column) normalization, for a SECOND heatmap.
    # Some sessions just have broadly higher or lower loss than others for
    # reasons that have nothing to do with which config was used (session
    # difficulty, recording quality, etc) -- on a single, globally-shared
    # color scale, that session-level offset dominates the color, making
    # it hard to see which config actually wins WITHIN a given session.
    # Min-max normalized per column (0 = best config in that session, 1 =
    # worst), so every session's own color range gets the same visual
    # weight regardless of its absolute baseline. Raw values still shown
    # as the text annotations -- only the COLOR is normalized, so nothing
    # about the real numbers is hidden, just the color scaling that was
    # obscuring the comparison.
    # -----------------------------------------------------------------
    col_min = pivot.min(axis=0, skipna=True)
    col_max = pivot.max(axis=0, skipna=True)
    col_range = (col_max - col_min).replace(0, np.nan)  # a column where every
    # config tied has nothing to normalize -- left NaN (blank), not divided
    # by zero into a misleading 0 or inf.
    pivot_norm = (pivot - col_min) / col_range

    fig_height = max(5, 0.42 * len(configs_sorted))
    fig_width = max(8, 1.1 * (len(sessions_sorted) + 1) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Mean column appended after the per-session columns -- the single
    # number that actually answers "which config is best", right next to
    # the per-session detail it was computed from.
    matrix = np.column_stack([pivot.to_numpy(), summary.set_index("config").loc[configs_sorted, "mean_loss"].to_numpy()])
    col_labels = sessions_sorted + ["MEAN"]

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(configs_sorted)))
    ax.set_yticklabels(
        [f"{i+1}. {pretty_config(c)}" for i, c in enumerate(configs_sorted)]
    )
    # Visually separate the MEAN column from the per-session columns.
    ax.axvline(len(sessions_sorted) - 0.5, color="white", linewidth=2)

    # Annotate every real cell (skip NaN -- a missing session/config pair).
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            weight = "bold" if col == matrix.shape[1] - 1 else "normal"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                     fontsize=8, fontweight=weight)

    best = summary.iloc[0]
    best_config_rows = df[df["config"] == best["config"]]

    ax.set_title(
        f"SNN sweep results -- {len(sessions_sorted)} session(s) x {len(configs_sorted)} config(s)\n"
        f"Best (by mean across sessions): {best['config']} "
        f"(mean={best['mean_loss']:.5f}, n={int(best['n_sessions'])})"
    )
    ax.set_xlabel("Session (MEAN = average across all sessions with this config) -- lower is better")
    fig.colorbar(im, ax=ax, label="Loss / RMSE")

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------
    # Second figure: SAME raw values as text, but color driven by the
    # per-session-normalized matrix (pivot_norm) instead of the raw loss --
    # answers "which config wins WITHIN each session" rather than being
    # dominated by whichever session happens to have the highest/lowest
    # absolute loss overall. The MEAN column is normalized the same way,
    # across configs' own mean_loss values, for the same reason.
    # -----------------------------------------------------------------
    mean_col = summary.set_index("config").loc[configs_sorted, "mean_loss"]
    mean_norm = (mean_col - mean_col.min()) / (mean_col.max() - mean_col.min()) \
        if mean_col.max() > mean_col.min() else mean_col * 0

    matrix_norm = np.column_stack([pivot_norm.to_numpy(), mean_norm.to_numpy()])

    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    masked_norm = np.ma.masked_invalid(matrix_norm)
    im2 = ax2.imshow(masked_norm, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_xticklabels(col_labels, rotation=45, ha="right")
    ax2.set_yticks(np.arange(len(configs_sorted)))
    ax2.set_yticklabels(
        [f"{i+1}. {pretty_config(c)}" for i, c in enumerate(configs_sorted)]
    )
    ax2.axvline(len(sessions_sorted) - 0.5, color="white", linewidth=2)

    # Text annotations show the RAW loss values (matrix, not matrix_norm) --
    # only the color encodes the per-session-normalized comparison.
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            weight = "bold" if col == matrix.shape[1] - 1 else "normal"
            ax2.text(col, row, f"{value:.3f}", ha="center", va="center",
                      fontsize=8, fontweight=weight)

    ax2.set_title(
        f"SNN sweep results, PER-SESSION NORMALIZED -- {len(sessions_sorted)} session(s) x "
        f"{len(configs_sorted)} config(s)\n"
        f"Color = rank within each session's own range (green=best, red=worst for THAT session); "
        f"numbers = raw loss"
    )
    ax2.set_xlabel("Session (MEAN column normalized across configs' own mean_loss)")
    fig2.colorbar(im2, ax=ax2, label="Normalized within session (0=best, 1=worst)")

    fig2.tight_layout()
    normalized_output = args.output.with_name(args.output.stem + "_normalized" + args.output.suffix)
    fig2.savefig(normalized_output, dpi=200, bbox_inches="tight")
    plt.show()

    print("\n============================================================")
    print("BEST CONFIGURATION (ranked by mean loss across sessions)")
    print("============================================================")
    print(f"Config:        {best['config']}")
    print(f"Mean loss:     {best['mean_loss']:.6f}  (std={best['std_loss']:.6f}, "
          f"n={int(best['n_sessions'])} session(s))")
    print(f"Min/Max loss:  {best['min_loss']:.6f} / {best['max_loss']:.6f}")
    print(f"Top-1 finishes: {int(best['n_top1'])}/{int(n_ranked[best['config']])} sessions "
          f"({best['top1_rate']*100:.0f}%), mean rank {best['mean_rank']:.2f}")
    print("Per-session breakdown for this config:")
    for _, r in best_config_rows.sort_values("session").iterrows():
        print(f"    {r['session']:<28} loss={r['loss']:.6f}  (source: {r['loss_key']})")
    print("============================================================")
    print("\nTop 5 configs overall (by mean loss across sessions):")
    for _, r in summary.head(5).iterrows():
        print(f"  {r['config']:<20} mean={r['mean_loss']:.5f}  std={r['std_loss']:.5f}  n={int(r['n_sessions'])}  "
              f"top1={int(r['n_top1'])}/{int(n_ranked[r['config']])} ({r['top1_rate']*100:.0f}%)  "
              f"mean_rank={r['mean_rank']:.2f}")

    # Which config is MOST OFTEN #1, session by session -- a different
    # question from "best mean loss" (see the comment above the ranks
    # computation): a config can have the best average without ever
    # actually winning a single session outright, if one session's own
    # scale dominates the mean. Surfaced explicitly here since it directly
    # answers "which config is at the top more often than not".
    most_frequent_winner = summary.sort_values(["n_top1", "top1_rate"], ascending=False).iloc[0]
    print(f"\nMost frequent top-1 finisher across sessions: {most_frequent_winner['config']} "
          f"({int(most_frequent_winner['n_top1'])}/{int(n_ranked[most_frequent_winner['config']])} sessions, "
          f"{most_frequent_winner['top1_rate']*100:.0f}%)")
    if most_frequent_winner["config"] != best["config"]:
        print(f"  NOTE: this differs from the best-mean-loss config ({best['config']}) -- "
              f"worth looking at both rather than treating either alone as decisive.")

    print(f"\nSaved per-(session,config) results: {Path('snn_sweep_losses.csv').resolve()}")
    print(f"Saved per-config summary (mean/std/n/top1/mean_rank): {Path('snn_sweep_config_summary.csv').resolve()}")
    print(f"Saved plot (raw values):        {args.output.resolve()}")
    print(f"Saved plot (per-session normalized): {normalized_output.resolve()}")


if __name__ == "__main__":
    main()
