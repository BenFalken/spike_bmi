"""
Build the 4x4 true-vs-decoded trajectory grid: a random sample of complete
trials from one session's test set, each showing the true cursor path and
every decoder's velocity-integrated reconstructed path, converging on (or
missing) the trial's target.

Reuses test_all_decoders.py's decoder-loading functions, COLORS dict, and
SNN full-test-set alignment machinery directly (import, not
reimplementation), so predictions and colors are guaranteed consistent with
that script's own comparison -- same models, same alignment scheme, same
color per decoder, no second source of truth to drift out of sync.

THIS REVISION follows test_all_decoders.py's rename (from
eval_all_decoders.py) and its run_snn_over_full_test_set() rework: the SNN
now runs across its ENTIRE available test set in ONE PASS -- with a
SINGLE, whole, continuous test trial (make_huge_dataset.py's own
test-saving convention, used throughout mua_8_group_uniform etc.), the
model resets exactly ONCE at the start and runs uninterrupted across the
entire stream; with the OLDER convention of many small, per-window test
trials, each trial is independently reset. Confirmed directly against
run_snn_over_full_test_set()'s own current source -- NOT the
checkpoint['args']['training_mode'] windowed/continuous dispatch an
earlier revision of this docstring described, which no longer exists
(train_bmi.py now has only one training mode; see its own module
docstring for why 'continuous'/'chunked' were tried and removed).
Either way, chunking into segment_samples-wide pieces for THIS figure's
own grid layout happens AFTER the full, uninterrupted prediction stream
is already computed -- purely a display choice, never a mid-stream
reset. Each SNN trial's own predictions align against ANN dense rows
using bmi.features.extract()'s lagged-target convention (see
calibrate_snn_ann_offset()/run_snn_over_full_test_set()'s own
docstrings for the full alignment reasoning).

Also changed: --min_trial_windows is replaced by --min_trial_duration_ms
(--trial_mode task only). Under the OLD coarser windowing a handful of
windows was a meaningful trial-completeness filter; under the current
dense (~4ms-step) ANN windowing a real trial spans tens to hundreds of
windows, so a fixed window-COUNT threshold goes stale every time
windowing density changes. Expressing the filter as a duration and
converting to a window count via step_time (read from the trial metadata
file, same as reconstruct_path() already does) keeps this meaningful
regardless of how dense the dataset is.

TWO --trial_mode options now, answering different questions:
  'task' (default): full task-trial reaches, bounded by real target
    changes -- REQUIRES --trials_filepath, make_trial_metadata.py's
    output, RE-RUN against the CURRENT dense ANN windowing
    (--wdw_time/--ol_time matching whatever produced --dataset_filepath).
  'segment': fixed-length, non-overlapping chunks of the comparison range
    (default 260 samples = 4x256ms=~1s, matching train_bmi.py's
    --truncation-chunks) -- needs NO trials file at all. Calls
    test_all_decoders.make_test_window_trajectory_grid() directly (the
    SAME function that script's own trajectory-grid figure uses), rather
    than reimplementing segment logic here or depending on a SEPARATE
    file (make_trial_metadata.py's old snn_trial_id, now removed) that
    would otherwise need regenerating every time the segment size changes
    anywhere in the project.

IMPORTANT CAVEAT (state this in the figure caption too): every decoder here
only ever predicts velocity, never position. Reconstructing a path means
cumulatively integrating predicted velocity across a trial, anchored to the
true position at trial onset. That integration compounds even small,
consistent per-step errors over a multi-second trial, so a decoder with an
unremarkable velocity RMSE can still visibly drift far from the true path by
a trial's end. This figure is illustrative of what using a decoder would
look like, not a second accuracy metric to read on its own -- the 4x2
figure's RMSE/CC remain the quantitative source of truth.

CLI usage (continuous test stream, chunked into ~2048ms segments purely
for display -- no resetting mid-stream, see above):
    python plot_trajectory_grid.py \
        --dataset_filepath /users/bfalkenb/data/bfalkenb/data/dataset/mua/indy_20160407_02_binning.h5 \
        --model_dir results/model_cache/indy_20160407_02 \
        --feature mua --test_frac 0.1 --decoders lstm,qrnn,snn \
        --snn_checkpoint_path checkpoints/bmi/mua_8_group_tausyn_sweep/indy_20160407_02/tausyn_8/best_model_weights.pth \
        --snn_dataset_path datasets/bmi/mua_8_group_uniform/indy_20160407_02 \
        --trial_mode segment --segment_samples 512 --n_trials 16 \
        --save_path results/indy_20160407_02_trajectory_grid.png
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from test_all_decoders import (
    ALL_DECODERS, DL_DECODERS, COLORS,
    load_dl_decoder, load_kf_decoder, load_wf_decoder,
    run_snn_over_full_test_set, calibrate_snn_ann_offset, compute_aligned_split,
    make_test_window_trajectory_grid,
)


def _infer_session_id(dataset_filepath):
    """indy_20160627_01_binning.h5 -> indy_20160627_01"""
    base = os.path.basename(dataset_filepath)
    for suffix in ('_binning.h5', '.h5'):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


# --------------------------------------------------------------------------- #
# Pure logic -- factored out so it can be tested without real data/models.
# --------------------------------------------------------------------------- #

def select_complete_trials(trial_id, n_common, min_windows=5, n_pick=16, seed=0):
    """Return up to n_pick (trial_id_value, i0, i1) tuples, i0/i1 inclusive
    window indices into the common (post-offset-trimming) index space.

    A trial "qualifies" only if every one of its windows lies strictly
    inside [0, n_common) -- i.e. it isn't cut off at the very start or end
    of the range every decoder can actually produce a prediction for -- and
    it has at least min_windows windows (drops degenerate, near-instant
    trials that wouldn't show a meaningful path).
    """
    trial_id = np.asarray(trial_id)
    candidates = []
    for tid in np.unique(trial_id):
        idx = np.where(trial_id == tid)[0]
        i0, i1 = idx.min(), idx.max()
        if i0 == 0 or i1 == n_common - 1:
            continue  # touches an edge of the common range -- likely truncated
        if (i1 - i0 + 1) < min_windows:
            continue
        candidates.append((tid, i0, i1))

    rng = np.random.default_rng(seed)
    if len(candidates) > n_pick:
        chosen_idx = rng.choice(len(candidates), size=n_pick, replace=False)
        candidates = [candidates[i] for i in sorted(chosen_idx)]
    return candidates


def reconstruct_path(anchor, velocities, step_time):
    """anchor: (2,) true position at trial onset.
    velocities: (m, 2) predicted velocity for each of the m windows after
    the anchor.
    Returns (m+1, 2): anchor followed by the cumulatively integrated path,
    so the output has the same length as the true path over
    [i0, i1] inclusive when velocities = pred[i0:i1] (m = i1 - i0 windows).
    """
    velocities = np.asarray(velocities)
    increments = velocities * step_time
    cum = np.cumsum(increments, axis=0)
    return np.vstack([anchor, anchor + cum])


def make_trajectory_grid_figure(trials, y_pos_common, target_pos_common, pred_common,
                                 step_time, session_id, n_trials_requested, colors=None,
                                 unit_label='trial', reconstruct=True, space_label='position'):
    """Pure plotting logic, given already-loaded/aligned arrays -- no file or
    model I/O here, so this can be exercised with synthetic data.

    trials: list of (trial_id, i0, i1) as returned by select_complete_trials
        (or any other (id, i0, i1) grouping -- e.g. test_all_decoders.py's
        fixed-length test-window segments, which have no natural single
        "target" the way a task trial does; see target_pos_common below).
    y_pos_common: (n_common, 2) true QUANTITY being plotted, aligned to the
        same index space as pred_common's arrays -- position when
        reconstruct=True, but see reconstruct below: with reconstruct=False
        this can be ANY 2D quantity (e.g. velocity itself), not just
        position specifically. Named y_pos_common for historical reasons
        (its original, and still most common, use).
    target_pos_common: (n_common, 2) target position, SAME index space, or
        None to skip the target-star marker entirely (e.g. for fixed-length
        segments that don't correspond to a single task trial with one
        well-defined target, or for a non-position space where "target"
        isn't a meaningful concept at all -- e.g. velocity-space).
    pred_common: {decoder_name: (n_common, 2) predicted quantity, same
        space as y_pos_common}.
    unit_label: word used in each panel's title ("trial 3 (65 windows)" by
        default) -- override to e.g. "segment" for non-task-trial groupings.
    reconstruct: True (default): pred_common holds VELOCITY, integrated via
        reconstruct_path() (anchored at each segment's true starting
        position) to get a comparable position-space curve -- the original
        use case. False: pred_common is ALREADY in the same space as
        y_pos_common (e.g. velocity plotted directly against velocity, no
        integration) -- plotted as-is. The only real difference this makes
        internally: with reconstruct=True, decoded curves are anchored to
        start exactly at the true curve's own first point (since integration
        needs a starting position and true position is the natural choice);
        with reconstruct=False there's no such anchoring -- each decoder's
        own first prediction is whatever it actually predicted, which is
        the more honest comparison for a space (like velocity) that isn't
        anchored to anything in the first place.
    space_label: word used in the title/axis description ("position" by
        default) -- override to e.g. "velocity" for a non-reconstructed,
        non-position space.
    """
    colors = colors if colors is not None else {}
    grid_side = int(np.ceil(np.sqrt(n_trials_requested)))
    fig, axes = plt.subplots(grid_side, grid_side,
                              figsize=(3.2 * grid_side, 3.2 * grid_side))
    axes = np.atleast_1d(axes).flatten()

    for ax, (tid, i0, i1) in zip(axes, trials):
        true_path = y_pos_common[i0:i1 + 1]
        anchor = true_path[0]

        ax.plot(true_path[:, 0], true_path[:, 1], color='black', linewidth=1.6,
                 label='true', zorder=5)
        for name, y_pred in pred_common.items():
            if reconstruct:
                decoded_path = reconstruct_path(anchor, y_pred[i0:i1], step_time)
            else:
                decoded_path = y_pred[i0:i1 + 1]  # already in the target space -- plot directly, no anchoring
            ax.plot(decoded_path[:, 0], decoded_path[:, 1],
                     color=colors.get(name, 'gray'), linewidth=1.0, alpha=0.85,
                     label=name.upper())
        ax.scatter(*anchor, marker='s', s=28, color='black', zorder=6,
                    label=f'{unit_label} start')
        ax.scatter(*true_path[-1], marker='o', s=40, facecolor='none',
                    edgecolor='black', linewidth=1.3, zorder=6,
                    label=f'true {space_label} end')
        if target_pos_common is not None:
            target = target_pos_common[i0]
            ax.scatter(*target, marker='*', s=150, color='black', edgecolor='white',
                        linewidth=0.7, zorder=7, label='target')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{unit_label} {tid} ({i1 - i0 + 1} windows)", fontsize=8)

    for ax in axes[len(trials):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    seen, uniq_handles, uniq_labels = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h); uniq_labels.append(l); seen.add(l)

    fig.suptitle(f"{session_id}: true vs. decoded hand {space_label} per {unit_label}",
                 fontsize=13, y=1.05)
    fig.legend(uniq_handles, uniq_labels, loc='upper center',
               ncol=min(len(uniq_labels), 8), bbox_to_anchor=(0.5, 1.00),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# --------------------------------------------------------------------------- #
# Real I/O -- loads actual data/models, calls the pure logic above.
# --------------------------------------------------------------------------- #

def main(args):
    print(f"Loading dataset: {args.dataset_filepath}")
    with h5py.File(args.dataset_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y_task = f['y_task'][()]
    y_pos_all = y_task[:, 0:2]
    y_vel_all = y_task[:, 2:4]
    N = X.shape[0]

    if args.trial_mode == 'task':
        print(f"Loading trial metadata: {args.trials_filepath}")
        with h5py.File(args.trials_filepath, 'r') as f:
            trial_id_all = f['trial_id'][()]
            target_pos_all = f['target_pos'][()]
            wdw_time = f.attrs['wdw_time']
            ol_time = f.attrs['ol_time']
        step_time = wdw_time - ol_time
        assert len(trial_id_all) == N, (
            f"trial metadata has {len(trial_id_all)} windows but dataset has {N} "
            f"-- re-run make_trial_metadata.py with matching --wdw_time/--ol_time "
            f"(this script assumes it was run against the CURRENT dense ANN "
            f"windowing -- see module docstring)")
    else:
        # 'segment' mode needs no trials file at all -- fixed-length
        # segments are computed directly by
        # test_all_decoders.make_test_window_trajectory_grid(), the SAME
        # function test_all_decoders.py's own trajectory-grid figure uses,
        # so there's no separate file to regenerate/keep in sync whenever
        # the segment size changes.
        trial_id_all = None
        target_pos_all = None
        # ANN's dense windowing step is this project's fixed native
        # sampling interval (4ms) -- same DELTA_TIME hardcoded throughout
        # make_dataset.py/make_snn_dataset.py/make_trial_metadata.py, not
        # something 'segment' mode needs a trials file to tell it.
        step_time = 0.004

    n_train, n_test = compute_aligned_split(N, args.test_frac)
    X_test = X[n_train:]
    y_pos_test = y_pos_all[n_train:]
    y_vel_test = y_vel_all[n_train:]
    if args.trial_mode == 'task':
        trial_id_test = trial_id_all[n_train:]
        target_pos_test = target_pos_all[n_train:]

    requested = [d.strip() for d in args.decoders.split(',')]
    unknown = set(requested) - set(ALL_DECODERS)
    if unknown:
        raise ValueError(f"Unknown decoder(s) {unknown}; choose from {ALL_DECODERS}")

    results = {}
    for decoder in [d for d in requested if d in DL_DECODERS]:
        print(f"Loading cached {decoder.upper()} model")
        y_pred, offset = load_dl_decoder(
            args.model_dir, args.feature, decoder, args.test_frac,
            X_test, y_vel_test, args.verbose)
        results[decoder] = (y_pred, offset)

    if 'kf' in requested:
        print("Loading cached KF model")
        results['kf'] = load_kf_decoder(
            args.model_dir, args.feature, args.test_frac, X_test, y_task[n_train:])

    if 'wf' in requested:
        print("Loading cached WF model")
        results['wf'] = load_wf_decoder(
            args.model_dir, args.feature, args.test_frac, X_test, y_vel_test)

    max_offset = max((offset for _, offset in results.values()), default=0)
    non_snn_end_candidates = [n_test] + [offset + len(y_pred) for y_pred, offset in results.values()]
    non_snn_end = min(non_snn_end_candidates)

    # SNN, if requested, determines the FINAL common range the same way
    # test_all_decoders.py's run_session() does: run_snn_over_full_test_set()
    # runs the SNN across its ENTIRE available test set (not a pre-selected
    # subset) and trims to the intersection with the other decoders'
    # available range -- always a single contiguous ANN row block, so it's
    # safe to let it override start_raw/n_common outright rather than
    # trying to separately intersect two independently-computed ranges.
    pred_common = {}
    if 'snn' in requested:
        if not (args.snn_checkpoint_path and args.snn_dataset_path):
            raise SystemExit("--decoders includes 'snn' but --snn_checkpoint_path/"
                              "--snn_dataset_path weren't both given.")
        print(f"Running SNN: checkpoint={args.snn_checkpoint_path}")
        ann_row_offset = calibrate_snn_ann_offset(args.snn_dataset_path, y_vel_test)
        snn_pred, snn_start_raw, snn_end_raw = run_snn_over_full_test_set(
            args.snn_checkpoint_path, args.snn_dataset_path, ann_row_offset=ann_row_offset,
            ann_start_offset=max_offset, ann_len=non_snn_end, verbose=args.verbose)
        if snn_pred is not None:
            start_raw, n_common = snn_start_raw, snn_end_raw - snn_start_raw
            pred_common['snn'] = snn_pred
        else:
            print("  SNN alignment produced no usable trials -- proceeding without it")
            start_raw = max_offset
            n_common = non_snn_end - max_offset
    else:
        start_raw = max_offset
        n_common = non_snn_end - max_offset

    y_pos_common = y_pos_test[start_raw:start_raw + n_common]

    for name, (y_pred, offset) in results.items():
        local_start = start_raw - offset
        pred_common[name] = y_pred[local_start:local_start + n_common]

    print(f"Common evaluable range: {n_common} windows across {len(pred_common)} decoders "
          f"(step_time={step_time * 1000:.2f} ms)")

    session_id = _infer_session_id(args.dataset_filepath)

    if args.trial_mode == 'segment':
        # No trials file, no select_complete_trials() filtering -- fixed-
        # length segments and how many FULL ones fit are both handled
        # internally by this function, identically to how
        # test_all_decoders.py's own trajectory-grid figure works (it's
        # the literal same function).
        make_test_window_trajectory_grid(
            session_id, y_pos_common, pred_common,
            segment_samples=args.segment_samples, n_segments=args.n_trials,
            step_time=step_time, save_path=args.save_path)
        return

    trial_id_common = trial_id_test[start_raw:start_raw + n_common]
    target_pos_common = target_pos_test[start_raw:start_raw + n_common]

    # Window-count threshold derived from a DURATION rather than a fixed
    # count -- see module docstring for why a fixed count goes stale as
    # soon as windowing density changes.
    min_windows = max(1, round((args.min_trial_duration_ms / 1000.0) / step_time))
    print(f"--min_trial_duration_ms={args.min_trial_duration_ms} -> "
          f"min_windows={min_windows} at this dataset's step_time")

    trials = select_complete_trials(
        trial_id_common, n_common, min_windows=min_windows,
        n_pick=args.n_trials, seed=args.seed)
    print(f"Selected {len(trials)} trials for the grid")
    if not trials:
        raise RuntimeError("No trials qualified -- check trial metadata / --min_trial_duration_ms.")

    fig = make_trajectory_grid_figure(
        trials, y_pos_common, target_pos_common, pred_common,
        step_time, session_id, args.n_trials, colors=COLORS)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--trials_filepath', type=str, default=None,
                         help="Required for --trial_mode task (make_trial_metadata.py output). "
                              "NOT needed for --trial_mode segment.")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--feature', type=str, default='sua')
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--decoders', type=str, default='mlp,lstm,qrnn,kf,wf,snn')
    parser.add_argument('--snn_checkpoint_path', type=str, default=None,
                         help="Path to this session's best_model_weights.pth "
                              "(e.g. checkpoints/bmi/mua/indy_20160627_01/best_model_weights.pth)")
    parser.add_argument('--snn_dataset_path', type=str, default=None,
                         help="Dir containing test/{i}.pkl for this session")
    parser.add_argument('--n_trials', type=int, default=16,
                         help="Number of panels shown -- task trials (task mode) or "
                              "segments (segment mode).")
    parser.add_argument('--trial_mode', type=str, default='segment', choices=['task', 'segment'],
                         help="'segment' (default): fixed-length, non-overlapping "
                              "--segment_samples-wide chunks of the comparison range, computed "
                              "directly with NO trials file needed -- the model runs across the "
                              "ENTIRE continuous test stream in one uninterrupted pass (no "
                              "mid-stream reset), then that already-computed prediction stream is "
                              "split into segments purely for this figure's own display -- calls "
                              "test_all_decoders.make_test_window_trajectory_grid() directly, so "
                              "this is guaranteed to match that script's own trajectory-grid "
                              "figure exactly (same function, not a reimplementation). 'task': "
                              "group by full task-trial reaches (trial_id) -- requires "
                              "--trials_filepath.")
    parser.add_argument('--segment_samples', type=int, default=512,
                         help="Only used with --trial_mode segment. Segment width in samples -- "
                              "default 512 = 512*4ms = ~2048ms, a purely visual chunking of the "
                              "already-computed, uninterrupted prediction stream (not a "
                              "resampling or a re-run of the model per segment).")
    parser.add_argument('--min_trial_duration_ms', type=float, default=500.0,
                         help='Only used with --trial_mode task. Minimum trial length, in ms, '
                              'to qualify for the grid (converted to a window count via this '
                              'dataset\'s own step_time -- replaces the old fixed '
                              '--min_trial_windows, which went stale under dense windowing; '
                              'see module docstring)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    if args.trial_mode == 'task' and not args.trials_filepath:
        raise SystemExit("--trial_mode task requires --trials_filepath "
                          "(make_trial_metadata.py output) -- or use --trial_mode segment, "
                          "which needs no trials file at all.")
    main(args)
