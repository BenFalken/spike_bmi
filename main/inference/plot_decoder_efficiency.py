"""
Build the "accuracy vs. computational cost" figure: one point per decoder,
x = per-sample inference latency, y = mean RMSE (already computed, pulled
from a multi-session combined_metrics.json), marker size = parameter count.

This exists to put a number on something the report's Introduction asserts
but never measures: that the SNN's motivation leans on hardware/compute
efficiency. Without this figure that claim is asserted, not shown.

THIS REVISION fixes two real problems, not just a rename:

1. eval_all_decoders.py was renamed to test_all_decoders.py, and
   predict_snn_window() no longer exists (renamed/restructured to
   predict_snn_trial()) -- imports/calls updated to match. load_snn_model()
   also changed shape: it now takes only checkpoint_path (no separate
   summary_path -- train_bmi.py's save_checkpoint() stores training args
   directly inside the checkpoint file itself now, so there's nothing
   separate to load), and returns (model, checkpoint, velocity_scale).
   --snn_summary_path is removed entirely as a result.

2. THE SNN TIMING METHODOLOGY WAS COMPARING THE WRONG THING. The previous
   version timed one predict_snn_window() call per TRIAL -- fine when
   trials were 256ms, but this project's settled dataset
   (make_large_snn_dataset.py's mua_large, 4x256ms=1024ms trials) makes
   that comparison meaningless: every OTHER decoder here is timed at
   4ms-per-sample resolution (the ANN's native dense-window step), so
   timing the SNN per 1024ms trial compares it against a task ~256x
   coarser than what it's actually doing, making it look artificially
   slow purely as a measurement artifact -- not a real cost difference.
   time_snn_per_timestep() now measures the SAME 4ms-per-sample
   granularity as everything else: the marginal cost of ONE additional
   T=1 forward() call, resetting only at each trial's own first timestep
   (matching train_bmi.py's windowed training exactly) and carrying state
   for every subsequent timestep within that trial -- which is also
   exactly how a real online decoder would actually be run, feeding one
   new 4ms spike-bin at a time rather than waiting for a full window to
   accumulate. This is the fair comparison this figure exists to make,
   and the reason to expect the SNN's numbers to look meaningfully
   different (likely better, given its sparse, event-driven per-timestep
   computation) now that it's measured on equal footing.

Two measurement choices worth being explicit about (and stating in the
figure's caption):

1. LATENCY IS SINGLE-SAMPLE, NOT BATCHED. Every decoder is timed by calling
   predict() on one window at a time, in a loop, not on a big batch. A real
   implant decodes one arriving window at a time -- batching would
   understate real deployment cost by amortizing per-call overhead across
   many samples in a way a real-time device never gets to. This makes
   LSTM/QRNN's numbers include real TensorFlow per-call Python/graph
   overhead, which is itself a genuine practical cost, not just an
   artifact -- but it does mean this isn't an idealized FLOPs comparison,
   and dedicated/embedded inference could look different. Say so in the
   caption.

2. PARAMETER COUNTS ARE NOT APPLES-TO-APPLES ACROSS DECODER FAMILIES. KF's
   count includes its full measurement-noise covariance matrix Q, which is
   (n_channels x n_channels) -- with ~96 MUA channels (this project's
   fixed channel count), that's ~9k scalars on its own, dwarfing its
   actual 6x6 dynamics matrices. That's a fitted statistic, not "model
   capacity" in the way a neural network's weight count is. Treat marker
   size as a rough visual cue, not a precise, comparable metric -- latency
   and RMSE (position on the axes) are the real story; size is secondary.
   Say this in the caption too.

This script writes its own minimal model loaders (rather than reusing
test_all_decoders.py's load_*, which are built to return predictions, not
model handles) since parameter counting needs the model object itself.

ENERGY, as of this revision, is measured HERE -- at the same per-sample
(or per-timestep, for SNN) granularity as latency -- rather than in
test_all_decoders.py, where an earlier revision added it. That turned out
to be the wrong place: test_all_decoders.py's load_kf_decoder / load_wf_decoder
/ load_dl_decoder call predict() ONCE on the ENTIRE session's test array
(and the SNN path loops over every trial in the dataset), so its
energy_j/inference_latency_s measured "cost of evaluating this whole
session's test set in one batched call," not "marginal cost of one
real-time sample arriving" -- a fundamentally different quantity, and NOT
apples-to-apples across decoders. Confirmed directly: comparing the two
pipelines' latency for the SAME session found ratios from ~40x (LSTM/QRNN,
where batching IS roughly linear in sample count) up to >1,000,000x for KF
-- span far too wide to be explained by "N samples vs. 1," which pointed at
KF/SNN's multi-row predict() paths scaling non-linearly with input length
inside test_all_decoders.py, on top of the batched-vs-single-sample
mismatch itself. This script's own single-sample timing loop (see point 1
below, and time_predict_per_sample() / time_snn_per_timestep()) was
already the validated, equal-footing measurement for latency -- reusing
the exact same loops for energy (via EnergyMeter, imported from
test_all_decoders.py) keeps both metrics answering the SAME question this
figure exists to answer: real-time, per-sample deployment cost, which is
also specifically what the report's SNN-efficiency claim is about (sparse,
event-driven PER-TIMESTEP computation -- see point 1). test_all_decoders.py
keeps its own (now-unused-by-this-figure) energy_j field for whoever wants
whole-test-set batch energy as an evaluation-harness-runtime number; that's
a legitimate but DIFFERENT question from this figure's.

Two measurement choices worth being explicit about (and stating in the
figure's caption):

1. LATENCY IS SINGLE-SAMPLE, NOT BATCHED. Every decoder is timed by calling
   predict() on one window at a time, in a loop, not on a big batch. A real
   implant decodes one arriving window at a time -- batching would
   understate real deployment cost by amortizing per-call overhead across
   many samples in a way a real-time device never gets to. This makes
   LSTM/QRNN's numbers include real TensorFlow per-call Python/graph
   overhead, which is itself a genuine practical cost, not just an
   artifact -- but it does mean this isn't an idealized FLOPs comparison,
   and dedicated/embedded inference could look different. Say so in the
   caption. ENERGY is measured the identical way, for the identical reason.

2. PARAMETER COUNTS ARE NOT APPLES-TO-APPLES ACROSS DECODER FAMILIES. KF's
   count includes its full measurement-noise covariance matrix Q, which is
   (n_channels x n_channels) -- with ~96 MUA channels (this project's
   fixed channel count), that's ~9k scalars on its own, dwarfing its
   actual 6x6 dynamics matrices. That's a fitted statistic, not "model
   capacity" in the way a neural network's weight count is. Treat marker
   size as a rough visual cue, not a precise, comparable metric -- latency
   and RMSE (position on the axes) are the real story; size is secondary.
   Say this in the caption too.

3. ENERGY MEASUREMENT WINDOW: RAPL's hardware counters (and the
   psutil-based proxy that's the fallback when RAPL isn't readable on a
   node -- see test_all_decoders.py's EnergyMeter) both need a long enough
   wall-clock window to produce a trustworthy reading. A handful of
   single-sample calls for a fast decoder like KF or WF (sub-millisecond
   each) isn't that window on its own, so energy is measured over a
   SEPARATE, LONGER combined pass (--n_energy_repeats full loops over the
   timing sample, back-to-back, default 10x the latency passes) rather
   than reused from the latency timing loop directly -- see
   time_predict_per_sample()'s own docstring for the full reasoning. This
   does mean profiling takes longer than it used to; --skip_energy is
   available if you just want latency/params quickly.

CLI usage:
    python plot_decoder_efficiency.py \
        --dataset_filepath ../data/bfalkenb/data/dataset/mua/indy_20160407_02_binning.h5 \
        --model_dir results/model_cache/indy_20160407_02 \
        --feature mua --test_frac 0.1 \
        --combined_metrics_path results/multi_session/combined_metrics.json \
        --decoders lstm,qrnn,kf,wf,snn \
        --snn_checkpoint_path checkpoints/bmi/mua_large_windowed/indy_20160407_02/best_model_weights.pth \
        --snn_dataset_path ./datasets/bmi/mua_large/indy_20160407_02 \
        --save_path results/decoder_efficiency.png \
        --profile_save_path results/decoder_efficiency_profiles/indy_20160407_02_profile.json
"""

import argparse
import json
import os
import pickle as pkl
import time

# MUST come before numpy/torch below -- energy_meter's own thread-pinning
# runs at ITS import time and needs to precede any BLAS-backed library's
# first import in this process to take effect (see energy_meter.py's own
# module docstring). This was actually being violated here before: the
# previous revision imported EnergyMeter lazily inside main(), well AFTER
# this module's own top-level `import torch`/`import numpy` above it --
# the pinning was silently a no-op for this script's whole profiling run.
from energy_meter import EnergyMeter, pin_torch_threads

import h5py
import numpy as np
import torch
pin_torch_threads()  # see pin_torch_threads()'s own docstring
import matplotlib.pyplot as plt


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


COLORS = {'lstm': 'darkorange', 'qrnn': 'seagreen',
          'kf': 'purple', 'wf': 'goldenrod', 'snn': 'royalblue',
          'speck': 'royalblue',  # SAME as "snn" -- speck is the chip-deployed
          'snn_pytorch': 'royalblue'}  # ALSO same -- infer_snn_speck.py's own "torch" run of
# the identical checkpoint (see plot_decoder_efficiency_aggregate.py's merge of it), the SAME
# underlying model as "snn"/"speck", not a separate decoder family; distinguished from snn
# visually by hatching (see HATCHES / make_efficiency_figure()) instead of a
# different hue, matching decoder_comparison_4x2.py's own convention.
HATCHES = {'speck': '///', 'snn_pytorch': '...'}  # dotted vs speck's crosshatch --
# decoder_comparison_4x2.py's own convention exactly: different patterns specifically because
# this pairing exists to let the two be compared directly (both come from the same
# infer_snn_speck.py run, of the same checkpoint, on the same sessions); identical hatching
# would defeat that.


# --------------------------------------------------------------------------- #
# Generic, duck-typed parameter counters -- take the loaded object itself,
# not a decoder class, so these have no heavy import dependencies and are
# testable with any stand-in object exposing the right attributes.
# --------------------------------------------------------------------------- #

def count_params_kf(kf_model):
    """kf_model.model is [A, W, H, Q] (see bmi.decoders.KalmanDecoder.fit)."""
    return int(sum(np.asarray(c).size for c in kf_model.model))


def count_params_sklearn(regressor):
    """Any sklearn-style linear regressor with .coef_ (and optionally
    .intercept_) -- covers WF's LinearRegression/Ridge/Lasso/ElasticNet."""
    n = np.asarray(regressor.coef_).size
    if hasattr(regressor, 'intercept_'):
        n += np.asarray(regressor.intercept_).size
    return int(n)


def count_params_keras(keras_model):
    return int(keras_model.count_params())


def count_params_torch(torch_model):
    """Any torch.nn.Module -- covers the SNN (SNN_Speck). Sums numel() over
    every registered nn.Parameter, which correctly EXCLUDES buffers
    (register_buffer), so the SNN's fixed `positions` lookup table (used for
    population-vector decoding, not learned) is not counted -- consistent
    with treating this as a model-capacity measure, not a raw tensor count."""
    return int(sum(p.numel() for p in torch_model.parameters()))


# --------------------------------------------------------------------------- #
# Timing -- single-sample loop, not batched. See module docstring point 1.
# --------------------------------------------------------------------------- #

def time_predict_per_sample(predict_one_fn, X_sample, n_repeats=3, n_warmup=3,
                             energy_meter_cls=None, n_energy_repeats=10):
    """predict_one_fn(x_row) must accept ONE sample (whatever shape that
    decoder needs) and return a prediction for it. Returns
    (latency_s, energy_j, energy_method):

    latency_s: median per-sample latency in seconds across n_repeats full
    passes over X_sample, after n_warmup untimed passes (critical for
    TF/Keras, which has first-call graph-tracing overhead that would
    otherwise dominate and badly overstate steady-state latency). This
    part is UNCHANGED from before energy measurement existed.

    energy_j / energy_method: None / None if energy_meter_cls is None
    (the default -- keeps this function import-light and testable with
    synthetic data when energy isn't needed, matching this module's
    existing lazy-import discipline for test_all_decoders -- see main()'s
    own local import). Otherwise energy_meter_cls should be
    test_all_decoders.EnergyMeter, passed in by the caller rather than
    imported at this module's top level.

    ENERGY IS MEASURED AS A SEPARATE, LONGER PASS, not derived from the
    n_repeats latency passes above -- two reasons:
      1. RAPL's hardware counters (and the psutil proxy EnergyMeter falls
         back to) both need a long enough wall-clock window to produce a
         trustworthy reading. A single n_repeats-scale pass over a fast
         decoder like KF or WF (sub-millisecond per call) may not be that
         window; running n_energy_repeats (default 10x more) back-to-back
         passes inside ONE EnergyMeter block gives more signal to work
         with, at the cost of extra profiling time.
      2. Keeping this separate leaves latency_s's own methodology (median
         of n_repeats independent passes, robust to one outlier pass)
         completely undisturbed.
    energy_j is normalized to PER-SAMPLE the same way latency_s is (total
    block energy / (n_energy_repeats * len(X_sample))).
    """
    for _ in range(n_warmup):
        for x in X_sample:
            predict_one_fn(x)

    pass_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        for x in X_sample:
            predict_one_fn(x)
        pass_times.append(time.perf_counter() - start)
    latency_s = float(np.median(pass_times)) / len(X_sample)

    energy_j, energy_method = None, None
    if energy_meter_cls is not None:
        with energy_meter_cls() as em:
            for _ in range(n_energy_repeats):
                for x in X_sample:
                    predict_one_fn(x)
        energy_method = em.energy_method
        energy_j = None if em.energy_j is None else em.energy_j / (n_energy_repeats * len(X_sample))
        if em.latency_s < 0.01:
            print(f"    [caution] energy measurement window was only {em.latency_s*1000:.2f} ms "
                  f"-- may be too short for a reliable {energy_method} reading; consider raising "
                  f"--n_energy_repeats if this decoder's energy number looks noisy/implausible "
                  f"across sessions")

    return latency_s, energy_j, energy_method


def time_snn_per_timestep(snn_model, snn_dataset_path, n_timing_samples, n_repeats=2, n_warmup=2,
                           energy_meter_cls=None, n_energy_repeats=10):
    """Times the SNN at the SAME granularity as every other decoder in
    this figure -- per-4ms-timestep latency, not per-trial. See module
    docstring for why timing per TRIAL (this project's mua_large trials
    are 1024ms) would badly overstate the SNN's real cost relative to
    decoders measured per 4ms sample: it isn't a fair comparison, it's a
    ~256x coarser task.

    Draws real timesteps sequentially from the SNN's own test .pkl
    trials (not the ANN's X_test -- those are the only genuine spike-
    raster input the SNN actually consumes), resetting ONLY at each
    trial's own first timestep (matching train_bmi.py's windowed
    training) and carrying state for every subsequent timestep within
    that trial. This is also exactly how a real online decoder would
    run: fed one new 4ms spike-bin at a time, not a whole pre-assembled
    window. Runs under torch.no_grad() throughout -- this is pure
    inference timing, and suppressing the reset (see run_pass()'s own
    comment for HOW, now that forward() no longer takes a reset_state
    argument at all) while accumulating gradients would be both wasted
    work and, left unchecked across many calls, a real memory growth
    problem.

    NOTE: the reset-suppression trick in run_pass() patches
    sinabs.utils.reset_states specifically -- it does NOT cover a
    --use-spikingjelly checkpoint's own functional.reset_net() path.
    This project's settled config uses the sinabs backend (neuron_type=
    'iaf', not --use-spikingjelly), so this hasn't been a real gap in
    practice, but a spikingjelly-backed SNN passed to this function
    would still reset every timestep despite the patch -- run_pass()
    checks and prints a loud warning rather than silently mismeasuring.

    Returns (latency_s, energy_j, energy_method) -- latency_s is the
    median per-TIMESTEP latency in seconds (n_repeats full passes over
    the same timestep sequence, after n_warmup untimed passes, UNCHANGED
    from before energy measurement existed). energy_j/energy_method: see
    time_predict_per_sample()'s docstring -- identical design (None/None
    when energy_meter_cls is None; otherwise a SEPARATE n_energy_repeats-
    pass block, normalized per-timestep the same way latency_s is).
    """
    test_dir = os.path.join(snn_dataset_path, 'test')
    trial_files = sorted((f for f in os.listdir(test_dir) if f.endswith('.pkl')),
                          key=lambda f: int(f.split('.')[0]))
    if not trial_files:
        raise FileNotFoundError(f"No .pkl trial files found under {test_dir}")

    timesteps = []  # [(input_row (C,), is_trial_start), ...]
    for fname in trial_files:
        with open(os.path.join(test_dir, fname), 'rb') as f:
            sample = pkl.load(f)
        input_spikes = sample['input_spikes']  # (C, T)
        for t in range(input_spikes.shape[1]):
            timesteps.append((input_spikes[:, t], t == 0))
        if len(timesteps) >= n_timing_samples:
            break
    timesteps = timesteps[:n_timing_samples]
    if len(timesteps) < n_timing_samples:
        print(f"  [note] only {len(timesteps)} timesteps available across all trials "
              f"under {test_dir} (requested {n_timing_samples})")

    _spikingjelly_warned = [False]  # list as a mutable cell -- warn once, not once per pass

    def run_pass():
        # model_bmi.py's own SNN_Speck.forward() no longer accepts
        # reset_state at all -- it now ALWAYS resets state unconditionally
        # at the start of every call (a deliberate, already-settled
        # training-time decision, not something to reopen here). But THIS
        # function's entire methodology depends on the opposite: state
        # persisting ACROSS separate per-timestep calls, resetting only at
        # each trial's own first timestep -- exactly how a real streaming
        # decoder would run, and (per this function's own docstring) the
        # SAME granularity every other decoder in this figure is timed
        # at. Silently batching the SNN into one per-trial call instead
        # would time it WITHOUT the per-call overhead every other decoder
        # still pays -- a real, direction-known bias favoring the SNN
        # specifically, not a fair comparison. So: suppress the reset
        # ourselves for non-trial-start timesteps, by temporarily
        # replacing sinabs's own reset_states with a no-op -- sinabs.utils
        # is the SAME module object model_bmi.py's own forward() calls
        # through (Python modules are singletons), so this patch actually
        # reaches it, scoped ONLY to this pass and guaranteed to be
        # undone even if something raises partway through.
        if getattr(snn_model, "use_spikingjelly", False) and not _spikingjelly_warned[0]:
            print("  WARNING: this SNN checkpoint uses --use-spikingjelly -- the reset-"
                  "suppression patch below only covers sinabs.utils.reset_states, NOT "
                  "spikingjelly's own functional.reset_net() path. Per-timestep state will "
                  "still be reset every call despite the patch, silently mismeasuring "
                  "latency for this specific checkpoint. Not currently fixed -- flag rather "
                  "than silently trust these numbers if you see this.")
            _spikingjelly_warned[0] = True

        import sinabs
        real_reset_states = sinabs.utils.reset_states
        try:
            with torch.no_grad():
                for x, is_trial_start in timesteps:
                    x_t = torch.from_numpy(x.astype(np.float32))[None, None, :]  # (T=1, N=1, C)
                    sinabs.utils.reset_states = real_reset_states if is_trial_start else (lambda *a, **k: None)
                    snn_model(x_t)
        finally:
            sinabs.utils.reset_states = real_reset_states

    for _ in range(n_warmup):
        run_pass()

    pass_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        run_pass()
        pass_times.append(time.perf_counter() - start)
    latency_s = float(np.median(pass_times)) / len(timesteps)

    energy_j, energy_method = None, None
    if energy_meter_cls is not None:
        with energy_meter_cls() as em:
            for _ in range(n_energy_repeats):
                run_pass()
        energy_method = em.energy_method
        energy_j = None if em.energy_j is None else em.energy_j / (n_energy_repeats * len(timesteps))
        if em.latency_s < 0.01:
            print(f"    [caution] SNN energy measurement window was only "
                  f"{em.latency_s*1000:.2f} ms -- may be too short for a reliable "
                  f"{energy_method} reading; consider raising --n_energy_repeats")

    return latency_s, energy_j, energy_method


# --------------------------------------------------------------------------- #
# Pure plotting logic -- given already-computed records, no I/O -- so this
# can be exercised with synthetic data.
# --------------------------------------------------------------------------- #

def make_efficiency_figure(records, colors=None, session_id=None, ax=None):
    """records: list of dicts, each with keys
        name, rmse, latency_s, param_count
    (energy_j/energy_method may also be present -- unused by this figure,
    which only plots latency/RMSE/param_count; see
    plot_decoder_efficiency_aggregate.py's make_energy_bar_figure() for
    the figure that uses them)

    ax: None (default) -- creates its OWN new Figure+Axes and returns the Figure, exactly as
        before this parameter existed; every EXISTING caller (this file's own single-session
        --save_path use, decoder_comparison_4x2.py-style callers) is completely unaffected.
        Pass an EXISTING Axes to draw into it instead -- no new Figure is created, no
        fig.tight_layout() is called (the caller owns the outer Figure and should call that
        once, after every Axes it's building has been drawn into), and the return value is
        that Axes' own Figure (ax.figure), not a new one -- added specifically so
        plot_decoder_efficiency_aggregate.py can put its Oscar-cohort and laptop-cohort
        panels side by side in ONE saved image (two plt.subplots() Axes) instead of two
        separate files, reusing this exact function's own scatter/legend/labeling logic for
        each panel rather than a second, hand-duplicated copy of it that could drift out of
        sync with this one over time.
    """
    colors = colors if colors is not None else COLORS
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    param_counts = np.array([r['param_count'] for r in records], dtype=float)
    max_params = param_counts.max() if len(param_counts) else 1.0
    # Marker AREA, not radius, should scale with param count (else visual
    # size differences get wildly exaggerated) -- sqrt here, then a floor so
    # the smallest decoder is still visible, and a cap so the largest
    # doesn't swallow the plot.
    sizes = 60 + 900 * np.sqrt(param_counts / max_params)

    for rec, size in zip(records, sizes):
        ax.scatter(rec['latency_s'] * 1000, rec['rmse'],
                   s=size, color=colors.get(rec['name'], 'gray'),
                   edgecolor='black', linewidth=0.8, alpha=0.85, zorder=3,
                   hatch=HATCHES.get(rec['name']))
        # Matches decoder_comparison_4x2.py's own _decoder_display_label() exactly -- same
        # three cases, so a decoder reads the same way in every figure this project produces.
        if rec['name'] == 'speck':
            label = "SNN *speck"
        elif rec['name'] == 'snn_pytorch':
            label = "SNN *PyTorch"
        else:
            label = rec['name'].upper()
        ax.annotate(label,
                     (rec['latency_s'] * 1000, rec['rmse']),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', fontsize=9, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Per-sample inference latency (ms, log scale)')
    ax.set_ylabel('Mean RMSE (aggregated across sessions)')
    title = 'Decoding accuracy vs. computational cost'
    if session_id:
        title += f'\n(latency/params from {session_id}; RMSE aggregated across sessions)'
    ax.set_title(title, fontsize=11)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.margins(y=0.18)  # headroom so the top annotation label isn't clipped

    # Small manual size legend (three reference sizes) rather than relying
    # on matplotlib's automatic legend_elements, since our sizes are already
    # sqrt-transformed and a literal legend of raw values would mislead.
    # Placed OUTSIDE the axes (below), not in a corner -- a corner legend
    # collides with whichever decoder happens to land there, and cheap-but-
    # inaccurate decoders (top-left: high RMSE, low latency) land in exactly
    # the most commonly-used legend corner often enough that this isn't a
    # rare edge case.
    ref_fracs = [0.1, 0.5, 1.0]
    # BUG FIX: was plt.scatter(...) -- the global pyplot function, which draws into whatever
    # plt.gca() ("current axes") happens to be, not necessarily THIS ax. Harmless in the old
    # ax=None-only usage (a freshly created Axes is always "current" immediately after
    # plt.subplots()), but wrong the moment an EXTERNAL ax is passed in and something else
    # (e.g. a second panel's own axes, created earlier in the same figure) is "current"
    # instead -- confirmed directly: these empty, label-only reference points landed on the
    # WRONG one of two side-by-side panels in a real test. ax.scatter(...), not plt.scatter(...),
    # ties this to the SAME Axes everything else in this function already draws into.
    handles = [ax.scatter([], [], s=60 + 900 * np.sqrt(f), color='gray',
                          edgecolor='black', alpha=0.6,
                          label=f'{int(f * max_params):,} params')
               for f in ref_fracs]
    ax.legend(handles=handles, title='Parameter count (rough scale)',
              loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8, title_fontsize=8, frameon=False)

    if owns_figure:
        # Only when THIS call created the figure -- a caller drawing into a shared, externally
        # -owned Axes (ax was passed in) is responsible for its own outer fig.tight_layout(),
        # called ONCE after every panel is drawn, not once per panel (repeated tight_layout()
        # calls on a multi-panel figure can fight each other over spacing).
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Real I/O
# --------------------------------------------------------------------------- #

def compute_mean_rmse(combined_metrics_path, decoders):
    with open(combined_metrics_path, 'r') as f:
        data = json.load(f)
    rmses = {d: [] for d in decoders}
    for entry in data.values():
        if not isinstance(entry, dict) or 'error' in entry:
            continue
        for d in decoders:
            if d in entry:
                rmses[d].append(entry[d]['rmse'])
    return {d: float(np.mean(v)) for d, v in rmses.items() if v}


def _infer_session_id(dataset_filepath):
    base = os.path.basename(dataset_filepath)
    for suffix in ('_binning.h5', '.h5'):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def main(args):
    from sklearn.preprocessing import StandardScaler  # noqa: F401 (scaler objects are unpickled)
    from bmi.decoders import QRNNDecoder, LSTMDecoder
    from test_all_decoders import load_snn_model  # EnergyMeter now imported at module top (see there)

    # None disables energy measurement entirely (--skip_energy) -- passed
    # through to time_predict_per_sample()/time_snn_per_timestep() below,
    # which treat energy_meter_cls=None as "skip, return (latency_s, None,
    # None)". EnergyMeter itself decides rapl vs. proxy_psutil once, at
    # test_all_decoders' own import time (just above, via `from
    # test_all_decoders import ...`) -- that choice is a property of THIS
    # NODE/PROCESS, so it's the same for every decoder profiled below, not
    # something to re-detect per decoder.
    energy_meter_cls = None if args.skip_energy else EnergyMeter

    session_id = _infer_session_id(args.dataset_filepath)
    print(f"Loading dataset: {args.dataset_filepath}")
    with h5py.File(args.dataset_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y_task = f['y_task'][()]
    N = X.shape[0]
    n_test = int(round(args.test_frac * N))
    X_test_raw = X[N - n_test:]

    n_timing_samples = min(args.n_timing_samples, len(X_test_raw))
    rng = np.random.default_rng(0)
    timing_idx = rng.choice(len(X_test_raw), size=n_timing_samples, replace=False)

    decoders = [d.strip() for d in args.decoders.split(',')]
    mean_rmse = compute_mean_rmse(args.combined_metrics_path, decoders)

    bundle_dir = os.path.join(args.model_dir, args.feature)
    records = []

    if 'kf' in decoders:
        print("Profiling KF")
        with open(os.path.join(bundle_dir, 'kf_model.pkl'), 'rb') as f:
            kf_model = pkl.load(f)
        with open(os.path.join(bundle_dir, 'kf_scaler.pkl'), 'rb') as f:
            kf_scaler = pkl.load(f)
        X_scaled = kf_scaler.transform(X_test_raw[timing_idx])
        # KF is inherently sequential (its own recursive state update), so
        # timing one predict() call per new sample is the faithful
        # real-time-equivalent measurement, not an approximation of it.
        y_init = y_task[N - n_test:][:1, :]
        latency, energy_j, energy_method = time_predict_per_sample(
            lambda x: kf_model.predict(x.reshape(1, -1), y_init), X_scaled,
            energy_meter_cls=energy_meter_cls, n_energy_repeats=args.n_energy_repeats)
        records.append({'name': 'kf', 'rmse': mean_rmse.get('kf'), 'latency_s': latency,
                         'param_count': count_params_kf(kf_model),
                         'energy_j': energy_j, 'energy_method': energy_method})

    if 'wf' in decoders:
        print("Profiling WF")
        with open(os.path.join(bundle_dir, 'wf_model.pkl'), 'rb') as f:
            wf_model = pkl.load(f)
        with open(os.path.join(bundle_dir, 'wf_scaler.pkl'), 'rb') as f:
            wf_scaler = pkl.load(f)
        with open(os.path.join(bundle_dir, 'wf_config.json'), 'r') as f:
            wf_config = json.load(f)
        timesteps = wf_config['timesteps']
        # WF needs `timesteps` consecutive raw rows per prediction; build
        # single-sample inputs by sliding a window ending at each timing index.
        X_scaled_full = wf_scaler.transform(X_test_raw)

        def wf_predict_one(end_idx):
            window = X_scaled_full[end_idx - timesteps + 1: end_idx + 1]
            flat = window.reshape(1, -1, order='F')
            return wf_model.predict(flat)

        valid_idx = timing_idx[timing_idx >= timesteps - 1]
        latency, energy_j, energy_method = time_predict_per_sample(
            wf_predict_one, valid_idx,
            energy_meter_cls=energy_meter_cls, n_energy_repeats=args.n_energy_repeats)
        records.append({'name': 'wf', 'rmse': mean_rmse.get('wf'), 'latency_s': latency,
                         'param_count': count_params_sklearn(wf_model.model),
                         'energy_j': energy_j, 'energy_method': energy_method})

    dl_classes = {'lstm': LSTMDecoder, 'qrnn': QRNNDecoder}
    for name, cls in dl_classes.items():
        if name not in decoders:
            continue
        print(f"Profiling {name.upper()}")
        with open(os.path.join(bundle_dir, f'{name}_config.json'), 'r') as f:
            config = json.load(f)
        with open(os.path.join(bundle_dir, f'{name}_scaler.pkl'), 'rb') as f:
            scaler = pkl.load(f)
        model = cls(config)
        X_scaled_full = scaler.transform(X_test_raw)

        # Both remaining DL decoders (MLP dropped project-wide -- see
        # module docstring) take a timesteps-long window per prediction;
        # the old branch handling MLP's non-windowed (single-row) input
        # shape is gone, not just unreachable.
        timesteps = config['timesteps']
        model.build(input_shape=(None, timesteps, config['input_dim']))
        model.load_weights(os.path.join(bundle_dir, f'{name}.weights.h5'))

        def predict_one(end_idx, m=model, ts=timesteps, X=X_scaled_full):
            window = X[end_idx - ts + 1: end_idx + 1][None, ...]
            return m.predict(window, verbose=0)

        valid_idx = timing_idx[timing_idx >= timesteps - 1]

        latency, energy_j, energy_method = time_predict_per_sample(
            predict_one, valid_idx, n_repeats=2, n_warmup=2,  # DL calls are slow; fewer repeats
            energy_meter_cls=energy_meter_cls, n_energy_repeats=args.n_energy_repeats)
        records.append({'name': name, 'rmse': mean_rmse.get(name), 'latency_s': latency,
                         'param_count': count_params_keras(model),
                         'energy_j': energy_j, 'energy_method': energy_method})

    if 'snn' in decoders:
        print("Profiling SNN")
        snn_model, checkpoint, velocity_scale = load_snn_model(args.snn_checkpoint_path, args.experiment)
        snn_model.eval()

        # Per-4ms-TIMESTEP latency, matching every other decoder in this
        # figure -- see time_snn_per_timestep()'s docstring and the module
        # docstring for why timing per TRIAL (this project's mua_large
        # trials are 1024ms) would badly overstate the SNN's real cost
        # relative to decoders measured per 4ms sample.
        latency, energy_j, energy_method = time_snn_per_timestep(
            snn_model, args.snn_dataset_path, args.n_timing_samples,
            n_repeats=2, n_warmup=2,
            energy_meter_cls=energy_meter_cls, n_energy_repeats=args.n_energy_repeats)
        records.append({'name': 'snn', 'rmse': mean_rmse.get('snn'), 'latency_s': latency,
                         'param_count': count_params_torch(snn_model),
                         'energy_j': energy_j, 'energy_method': energy_method})

    records = [r for r in records if r['rmse'] is not None]
    print("Records:")
    for r in records:
        energy_str = (f"energy={r['energy_j']*1e6:.3f} uJ/sample ({r['energy_method']})"
                      if r.get('energy_j') is not None
                      else f"energy=n/a ({r.get('energy_method') or 'skipped'})")
        print(f"  {r['name']:>5s} | RMSE={r['rmse']:.2f} | "
              f"latency={r['latency_s']*1000:.4f} ms/sample | params={r['param_count']:,} | "
              f"{energy_str}")

    if args.profile_save_path:
        # Latency, param_count, AND energy_j -- all measured at the SAME
        # per-sample granularity by this script, so they belong together.
        # RMSE remains deliberately excluded: it's already aggregated
        # separately, from combined_metrics.json, by whatever aggregate
        # script reads these files (see plot_decoder_efficiency_aggregate.py)
        # -- duplicating it here would just be a second, redundant copy of
        # the same numbers to keep in sync.
        #
        # energy_method is stored ONCE per session (top-level), not per
        # decoder -- it's a property of THIS node/process (whether RAPL
        # counters were readable), not of any individual model, so every
        # decoder profiled in this one invocation shares the same value.
        # None if --skip_energy was set (no decoder has energy_j either).
        session_energy_method = next((r['energy_method'] for r in records
                                       if r.get('energy_method') is not None), None)
        os.makedirs(os.path.dirname(args.profile_save_path) or '.', exist_ok=True)
        profile = {'session': session_id,
                   'energy_method': session_energy_method,
                   'decoders': {r['name']: {'latency_s': r['latency_s'],
                                             'param_count': r['param_count'],
                                             'energy_j': r.get('energy_j')}
                                for r in records}}
        with open(args.profile_save_path, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"Saved per-session profile to {args.profile_save_path}")

    if args.skip_figure:
        print("--skip_figure set -- not generating the single-session comparison figure "
              "(used when profiling many sessions in a batch specifically to feed "
              "plot_decoder_efficiency_aggregate.py; that script produces the figure that "
              "actually matters for that use case, not this one).")
        return

    fig = make_efficiency_figure(records, session_id=session_id)
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filepath', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--feature', type=str, default='mua')
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--combined_metrics_path', type=str, required=True)
    parser.add_argument('--decoders', type=str, default='lstm,qrnn,kf,wf,snn')
    parser.add_argument('--snn_checkpoint_path', type=str, default=None)
    parser.add_argument('--snn_dataset_path', type=str, default=None)
    # BUG FIX: load_snn_model() (test_all_decoders.py's own -- see that file for why
    # `experiment` has no default there) has always required this positional argument, but
    # nothing in this script ever defined it or passed it through -- the call site below used
    # to be load_snn_model(args.snn_checkpoint_path) alone, which raised
    # "TypeError: load_snn_model() missing 1 required positional argument: 'experiment'" the
    # moment SNN profiling was actually reached (confirmed on a real sbatch run, loco
    # session). default='bmi' rather than required=True, deliberately: this project's own
    # sbatch array jobs already build an 'experiment' shell variable per task (visible in a
    # real log as "Array task N -> experiment=bmi subject=...") but that log shows it was
    # NEVER forwarded as a --experiment flag to this script -- making the flag required would
    # keep failing every array task exactly as before until the sbatch script is ALSO
    # updated to pass --experiment "$experiment". A default of 'bmi' unblocks every session
    # this project has actually run through this script so far (every real invocation seen
    # has been experiment=bmi) without requiring that separate change first. It is silently
    # WRONG the moment this script is ever pointed at an hkm checkpoint/dataset without
    # --experiment hkm passed explicitly -- update the sbatch script to pass
    # --experiment "$experiment" so this default is never actually relied on in practice.
    parser.add_argument('--experiment', type=str, default='bmi', choices=['bmi', 'hkm'],
                         help="Passed to load_snn_model() -- selects models.model_bmi vs "
                              "models.model_hkm. Default 'bmi' matches every session this "
                              "script has been run on so far; pass explicitly for hkm.")
    parser.add_argument('--n_timing_samples', type=int, default=50,
                         help='How many single-sample predict() calls per LATENCY timing pass '
                              '(and, for SNN, how many individual 4ms TIMESTEPS -- not '
                              'trials -- per timing pass). Kept modest by default since DL '
                              'single-sample calls are slow.')
    parser.add_argument('--n_energy_repeats', type=int, default=10,
                         help='How many full n_timing_samples-sized passes to run back-to-back '
                              'inside ONE energy-measurement window -- separate from (and, by '
                              'default, 10x longer than) the latency timing passes, since RAPL '
                              '(or the psutil proxy) needs a longer window than a handful of '
                              'single-sample calls to produce a trustworthy reading, especially '
                              'for fast decoders like KF/WF. Raise this if energy numbers look '
                              'noisy across sessions; see time_predict_per_sample()\'s docstring.')
    parser.add_argument('--skip_energy', action='store_true',
                         help='Skip energy measurement entirely (latency/params only) -- faster '
                              'profiling when energy isn\'t needed this run.')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--profile_save_path', type=str, default=None,
                         help='Optional: save this session\'s raw {decoder: {latency_s, '
                              'param_count, energy_j}} (plus a session-level energy_method) to '
                              'this JSON path -- for aggregating across many sessions, see '
                              'plot_decoder_efficiency_aggregate.py.')
    parser.add_argument('--skip_figure', action='store_true',
                         help='Skip generating the single-session comparison figure -- use '
                              'when profiling many sessions in a batch (--profile_save_path) '
                              'and only the aggregate figure across all of them matters.')
    args = parser.parse_args()
    main(args)