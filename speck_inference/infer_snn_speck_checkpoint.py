"""
Inference script for the current BMI SNN_Speck model (model_bmi.py) --
step 1 of building toward dynapcnn conversion and Speck2f deployment.
Ported from an older script written against a different, image-based
pipeline (models.model_new: Conv2d/pool_type/speck_compatible, 2D
position decoding on a fixed canvas) -- this is a real rewrite against
this project's actual architecture and data, not a search-and-replace of
import paths.

WHAT ACTUALLY CHANGED (not exhaustive renaming -- these are real,
substantive differences between the two pipelines):

1. MODEL LOADING IS NOW A SINGLE, ALREADY-TESTED FUNCTION CALL. The old
   script manually looped over Conv2d/Linear weight-norm
   parametrizations, then separately copied spike_threshold/min_v_mem
   per layer, then separately copied _temporal_decay_raw (with a real,
   documented bug: that copy was missing entirely in an earlier version,
   silently running eval with the untrained default decay instead of the
   trained one). None of that manual loading exists here: this
   architecture doesn't use weight-norm parametrization at all (plain
   nn.Linear, no Conv2d anywhere), and spike_threshold/_temporal_decay_raw/
   positions/neutral_scaled_pred are all real, registered
   parameters/buffers that a standard state_dict load already restores
   correctly. test_all_decoders.py's load_snn_model() already does this
   -- including the correct backward-compatible handling of checkpoints
   that predate neutral_scaled_pred -- so it's reused directly rather
   than reimplemented a third time.

2. decode_from_ema() now matches SNN_Speck.decode_output() EXACTLY,
   including its zero-spike fallback. The old script's version used an
   unconditional denominator clamp (`.clamp(min=1)` with no fallback
   branch) -- exactly the pre-fix bug this project already found and
   fixed in decode_output() itself (see model_bmi.py's own history): a
   sample with zero accumulated spikes in an axis silently decoded to
   scaled 0.0, which unscales to a value beyond the trained velocity
   range entirely, not a neutral "no signal" prediction. This version
   pulls n_bins/positions/neutral_scaled_pred straight off the loaded
   model (never hardcoded) specifically so this reimplementation can't
   silently drift out of sync with the real one the way the old one did.

3. The `dynapcnn`-prep path (snn_seq) now explicitly asserts
   use_iaf_squeeze=True before running. _SinabsNeuronLayer.forward()
   only calls its wrapped neuron directly (matching what snn_seq's plain
   nn.Sequential does) when the layer was built as a Squeeze variant --
   otherwise the wrapper's own unsqueeze(1)/squeeze(1) time-dimension
   handling is required and snn_seq, which bypasses that wrapper
   entirely, would silently feed the wrong shape into a non-Squeeze
   layer. True for this project's actual training convention
   (--use-iaf-squeeze is always passed) and also a real, independent
   requirement of Speck deployment itself, not just a script
   compatibility concern -- so this assertion isn't a workaround, it's
   confirming a requirement that would need to hold regardless.

4. Evaluation is now in PHYSICAL velocity units (via train_bmi.py's
   unscale_velocity(), using velocity_lo/hi/margin read from the
   checkpoint), matching train_bmi.py's/test_all_decoders.py's own
   convention -- not the old script's raw scaled-space MSE, which isn't
   directly interpretable as a velocity error.

5. NEW: model() and snn_seq() are cross-checked against each other on
   the same real trial and asserted to agree closely -- the old script
   never did this despite building both paths, and disagreement between
   the two forward-pass mechanics (BEFORE dynapcnn conversion is even
   in the picture) would be worth catching immediately rather than
   discovering later, once actual hardware is in the loop and harder to
   debug against.

6. Visualization is now a RECONSTRUCTED HAND-POSITION animation (via
   plot_trajectory_grid.py's reconstruct_path(), integrating the
   decoded velocity from the trial's own true starting position), not
   the old script's fixed-canvas 2D image target-tracking. This is a
   real, stated assumption, not an obvious translation: the old
   pipeline's task decoded absolute (x, y) position on a fixed frame
   directly; this pipeline decodes VELOCITY. Reconstructed position
   (matching what test_all_decoders.py's own trajectory-grid figure
   already does, reusing the same tested function) is the natural
   analogue for "does the predicted movement track the real one," but
   it's a design choice worth confirming rather than assuming silently
   -- if a different visualization is actually wanted here, say so.

WHAT THIS SCRIPT DOES NOT DO YET (deliberately -- this is step 1 of
several, per your own framing): no DynapcnnNetwork construction, no
specksim/on-chip execution, no quantization/calibration for actual
Speck2f deployment. snn_seq here is the CPU-side stand-in that will
eventually get handed to DynapcnnNetwork, not a working chip pipeline
itself -- that's the next step to build once this foundation is
confirmed correct.

7. ENERGY TRACKING, added to put the deployment story this script exists
   for on an actual measured footing, not just latency. THREE genuinely
   DIFFERENT kinds of number, kept clearly separate rather than
   pretending they're the same kind of thing:

   - torch/discretized: a MAC/ACC-based ESTIMATE (method='mac_acc_estimate'),
     via op_energy_estimate.py's finalize_snn_ops() -- hardware-agnostic
     effective-operation counts (from model_bmi.py's own count_ops=True
     path on the original model object, a separate call from the
     run_torch_model()/snn_seq() comparison path itself, since the two
     are already cross-checked to agree) converted to a [low, high]
     joules range via reference per-op energy figures for a REFERENCE
     process node. Answers "how does this computation scale
     algorithmically," NOT "what did this specific run on this specific
     machine actually cost" -- replaced the older EnergyMeter-based host-
     CPU approximation for these two implementations specifically, so
     they're now directly comparable to the same MAC/ACC methodology
     test_all_decoders.py already uses for every other decoder.
   - specksim still runs as ordinary Python on THIS machine's CPU -- its
     energy is HOST energy, via EnergyMeter (imported from
     energy_meter.py -- a small, dependency-light module containing just
     this class plus thread-pinning, factored out of test_all_decoders.py
     specifically so this script isn't dragging in that module's full
     machinery -- torch, h5py, sklearn, bmi.decoders, sinabs, plus its
     own CUDA_VISIBLE_DEVICES-forcing -- for one class. See
     energy_meter.py's own module docstring for the RAPL-vs-proxy
     mechanics). Importing energy_meter also pins math-library thread
     counts (OMP_NUM_THREADS/etc., plus an explicit pin_torch_threads()
     call right after `import torch` below) for the same equal-footing
     reasons as plot_decoder_efficiency.py -- THIS IMPORT MUST STAY
     BEFORE `import torch`/`import numpy` a few lines down: an earlier
     revision imported EnergyMeter (then still inside test_all_decoders.py)
     AFTER those imports, which silently made the thread-pinning a no-op
     for this script's entire run. See energy_meter.py's own
     import-order warning for the general version of this bug. torch/
     discretized's own decode loops still use EnergyMeter too, but ONLY
     for latency_s now, not energy_j -- see run_torch_model()'s own
     docstring.
   - speck is REAL ON-CHIP energy, read from the devkit's own
     PowerMonitor telemetry (actual current/power draw on the physical
     ASIC) and integrated over each trial's run -- see
     integrate_power_events()'s docstring. This is the number the whole
     deployment effort is actually for.

   Comparing host-software energy against dedicated-ASIC energy IS the
   comparison this deployment work exists to make (that's the entire
   point of moving inference onto neuromorphic hardware) -- but the
   summary at the end says explicitly, every time it's printed, that
   these are two different measurement mechanisms, not the same ruler
   applied twice. UPDATE: the power-monitoring wiring/units below are now
   VERIFIED against sinabs' own official power-monitoring guide
   (https://sinabs.readthedocs.io/v2.0.3/speck/notebooks/
   power_monitoring.html), not just researched-but-unverified as this
   point originally said -- that verification caught two real bugs
   (wrong event source: power_monitor has no .get_events() at all, the
   events come from a separate buffer node wired into the graph; and
   wrong units: .value is in watts, not milliwatts as first assumed).
   See integrate_power_events()'s own docstring for the specifics.
"""

import argparse
import os
import time  # moved up from its old location deep in section 13 -- run_torch_model()
import json  # for the per-session metrics.json export (section 14b)
# (via progress_checkpoint(), added for mua_combined's long single-stream runs) now
# needs this at the cross-check call in section 6, well before section 13 used to
# import it -- a genuine ordering bug this fixes, not just tidiness

# Default this script to a SINGLE thread, deliberately, BEFORE energy_meter
# is imported below (which reads this env var at ITS OWN import time -- see
# its module docstring). This is a real methodological choice, not just
# performance: torch/BLAS freely using every core on the machine is exactly
# what inflated the earlier "383.6 J" Torch energy reading ~11x (see
# implied_cores in the ENERGY SUMMARY) -- pinning to 1 thread makes the
# host-side energy number reflect a single, controlled, reproducible
# core's worth of work, closer to what "the CPU cost of running this model"
# should mean for a fair comparison against Speck's dedicated hardware.
# setdefault (not a hard overwrite) so an explicit `export
# ENERGY_METER_NUM_THREADS=N` before running this script still wins, if you
# deliberately want a different thread count for a specific comparison.
os.environ.setdefault("ENERGY_METER_NUM_THREADS", "1")

# MUST come before `import torch`/`import numpy` below -- energy_meter's
# own thread-pinning runs at ITS import time and needs to precede any
# BLAS-backed library's first import in this process to take effect (see
# energy_meter.py's own module docstring). THIS WAS A REAL BUG in the
# previous revision of this file: EnergyMeter was imported from
# test_all_decoders.py, but that import statement sat AFTER `import
# torch`/`import numpy` a few lines above it -- the thread-pinning was
# silently a no-op for this script's entire run, the whole time. Also
# switched from test_all_decoders.py to the standalone energy_meter.py:
# this script only ever needed EnergyMeter, not test_all_decoders.py's
# full machinery (torch, h5py, sklearn, bmi.decoders, sinabs, plus that
# module's own CUDA_VISIBLE_DEVICES-forcing, which this script never
# asked for and doesn't want silently applied to it).
from energy_meter import EnergyMeter, pin_torch_threads, ASSUMED_CPU_TDP_WATTS
from op_energy_estimate import finalize_snn_ops  # MAC/ACC-based energy estimate for
# torch/discretized -- replaces the old EnergyMeter-based host-CPU approximation for
# those two implementations specifically (see run_torch_model()'s own docstring);
# EnergyMeter itself stays in use for latency timing and for specksim's own energy

import numpy as np
import torch
pin_torch_threads()  # see pin_torch_threads()'s own docstring for why this needs
# to happen right after `import torch`, before torch is used anywhere else below
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models.model_bmi import _SinabsNeuronLayer, _SJNeuronLayer
from snn_inference_utils import load_snn_model, reconstruct_path

_MODEL_ORDER = ["torch", "discretized", "specksim", "speck"]  # canonical order for
# printouts/plots, independent of the order --models was typed in

IMPL_COLORS = {"torch": "royalblue", "discretized": "seagreen",
               "specksim": "darkorange", "speck": "crimson"}  # single source of truth
# for every plot in this script that colors by implementation -- previously
# redefined identically in plot_comparison()/plot_velocity_over_time(), now
# reused in plot_energy_summary() too

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--experiment", type=str, default="bmi", choices=["bmi", "hkm"],
    help="Which model module to load the checkpoint through: 'bmi' -> "
         "models.model_bmi, 'hkm' -> models.model_hkm (a separate fork -- see that "
         "file's own docstring for why). Default 'bmi' matches this script's own "
         "existing default --session-id/--checkpoint-path/--dataset-path "
         "conventions; pass --experiment hkm explicitly for an HKM checkpoint.")
parser.add_argument(
    "--subject", type=str, default=None,
    help="Which subject's velocity scalers (velocity_scalers.json) to use for this "
         "session's dataset -- e.g. 'indy', 'loco'. Passed straight to "
         "create_dataloaders(experiment=..., subject=...). Default None -> dataset.py "
         "tries to derive it from --dataset-path's own '.../{experiment}/{subject}/...' "
         "segment, which works for the Oscar layout (snn_datasets/bmi/indy/mua_8_group/"
         "{session}) but NOT for a layout with no subject directory level, such as this "
         "machine's datasets/bmi/mua/{session} -- pass --subject explicitly there "
         "(run_speck_inference_sweep.sh does this automatically).")
parser.add_argument(
    "--models", nargs="+", default=["torch", "speck"],
    choices=_MODEL_ORDER,
    help="Which implementations to run and compare (default: torch discretized "
         "specksim). 'discretized'/'specksim'/'speck' all require building the "
         "DynapcnnNetwork conversion (section 9) -- selecting only 'torch' skips that "
         "entirely, for a fast, no-devkit-needed sanity check. Selecting 'speck' "
         "requires the physical Speck2f devkit to be connected on THIS machine -- this "
         "replaces the old separate infer_snn_speck_hardware.py file: run with "
         "'--models torch discretized specksim speck' on the machine connected to the "
         "chip instead.")
parser.add_argument(
    "--session-id", type=str, default=None,
    help="Which session's checkpoint/data to run. Used to build default "
         "--checkpoint-path/--dataset-path if those aren't given explicitly. Default "
         "None -> derived from --dataset-path's own last path component (or, failing "
         "that, --checkpoint-path's parent directory name); falls back to this script's "
         "original default, indy_20160407_02, only if neither path was given either.")
parser.add_argument(
    "--override-tau-syn", type=float, default=None,
    help="DIAGNOSTIC ONLY -- overrides EVERY layer's own tau_syn (torch's own model "
         "object, BEFORE snn_seq/snn_disc/dynapcnn_net are built from it -- see the "
         "override's own placement, right after the checkpoint loads, for why this "
         "reaches all three implementations from one shared source rather than "
         "drifting them further apart). Default None -- use the checkpoint's own "
         "trained tau_syn, unmodified; this is the normal, accuracy-preserving path. "
         "Set to isolate whether tau_syn's own real-time decay assumption (calibrated "
         "for delta_time=0.004s per step) being mismatched against speck's ACTUAL "
         "wall-clock rate (--speck-wait-time, which is NOT 4ms) explains speck's "
         "own accuracy gap -- e.g. --override-tau-syn 1.0 makes synaptic current "
         "decay almost fully within one step regardless of how long that step "
         "actually took in real time, removing the mismatch as a variable rather "
         "than trying to match it. WILL degrade torch/discretized accuracy too, on "
         "purpose -- this is testing whether all three implementations move CLOSER "
         "together under matched conditions, not testing which one is 'right'.")
parser.add_argument(
    "--speck-wait-time", type=float, default=0.001,
    help="DIAGNOSTIC/TUNING -- real wall-clock seconds run_speck_model() waits after "
         "sending one timestep's input before reading output spikes and sending the "
         "next. Default 0.001 (1ms) matches this file's own long-standing default. "
         "NOTE: this is NOT delta_time (0.004s, the project's own native step "
         "duration the model was trained on) -- they have always been two different "
         "numbers; --override-tau-syn above is what actually reconciles that "
         "mismatch, not this value alone (confirmed directly: raising this alone "
         "to 0.005 made speck's own accuracy WORSE, not better -- see the "
         "conversation this flag came from for the full experiment).")
parser.add_argument(
    "--speck-raster-dt", type=float, default=0.1,
    help="DIAGNOSTIC/TUNING -- dt passed to chip_factory.raster_to_events() when "
         "converting one timestep's input spike counts into chip-bound events. "
         "Default 0.1, this file's own long-standing default. NOTE: confirmed "
         "directly (not assumed) that changing this alone, with --speck-wait-time "
         "held at its own default, produced no measurable difference in speck's own "
         "results at all -- kept configurable for completeness/future experiments, "
         "not because it's known to matter.")
parser.add_argument(
    "--checkpoint-path", type=str, default=None,
    help="Path to this session's best_model_weights.pth. Default: "
         "./checkpoints/bmi/mua/{session-id}/best_model_weights.pth")
parser.add_argument(
    "--dataset-path", type=str, default=None,
    help="Path to this session's mua_combined test data directory. Default: "
         "./datasets/bmi/mua_combined/{session-id}")
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="Where to save this session's GIFs/plots/speck_results.json. Default: BMI_GIFS "
         "(matching the original, single-session convention) -- pass this explicitly "
         "when running multiple sessions so each one's outputs land in its own "
         "directory rather than overwriting the last session's.")
parser.add_argument(
    "--idle-baseline-seconds", type=float, default=0.0,
    help="OPT-IN, default 0 (off; a sweep that doesn't pass it behaves exactly as before). If > 0, the chip's power "
         "is measured for this many seconds with the network configured but NO input events (v_mem zeroed, "
         "same graph), immediately before the real run, and saved as power_audit.idle_mean_power_w with "
         "dynamic power = active - idle. This is the sinabs power-monitoring guide's own idle-then-dynamic "
         "procedure, and the only way to separate what the WORKLOAD costs from the chip's static floor -- "
         "which is ~all of the measured draw for this project's tiny network (see summarize_power_events()). "
         "The idle floor is a property of the chip configuration, not of a session, so measuring it on one "
         "or two sessions (10-30 s each) is enough; aggregate_speck_results.py --idle-power-w can then apply it "
         "to every session.")
parser.add_argument(
    "--no-plots", action="store_true",
    help="Skip EVERY figure and animation -- the three GIFs (prediction_*, comparison_*, "
         "crosshair/velocity_crosshair_*), velocity_over_time_*.png and energy_summary.png -- and "
         "the frame reconstruction that only feeds them. Nothing that feeds speck_results.json is "
         "affected. On a real session these accounted for roughly 175 s of a ~5 min run (about 26 s "
         "inside the snn_seq evaluation plus ~150 s after the chip loop), so a sweep with this set "
         "runs about 2-3x faster. Also enabled by setting the environment variable SPECK_NO_PLOTS=1, "
         "which needs no change to run_speck_inference_sweep.sh (child processes inherit it).")
args = parser.parse_args()

# --no-plots OR SPECK_NO_PLOTS=1 (any value except empty / 0 / false / no). The environment variable
# exists so a sweep script that launches this file can be switched without editing it.
NO_PLOTS = bool(args.no_plots) or os.environ.get("SPECK_NO_PLOTS", "").strip().lower() not in ("", "0", "false", "no")
if NO_PLOTS:
    print("Plots / GIFs     : SKIPPED (--no-plots or SPECK_NO_PLOTS) -- speck_results.json is written as usual")
if "speck" in args.models:
    # Printed AND saved (see speck_wait_time_s in speck_results.json) because this setting is invisible
    # everywhere else: a 4 ms run and a 1 ms run of the same session differ only in per-step latency and the
    # silent-step fraction, and a sweep mixing the two would look fine until someone compared latencies.
    print(f"Speck readout    : wait {args.speck_wait_time * 1e3:g} ms per step, raster dt {args.speck_raster_dt:g} "
          f"(defaults: 1 ms / 0.1)")

# BUG FIX: SESSION_ID used to be `args.session_id` unconditionally, which defaulted to
# "indy_20160407_02" -- and run_speck_inference_sweep.sh never passes --session-id (it
# passes --checkpoint-path/--dataset-path explicitly instead), so every session in a sweep
# printed "Session: indy_20160407_02" and, worse, wrote `"session_id": "indy_20160407_02"`
# into its own speck_results.json regardless of which session actually ran (confirmed from a
# real sweep log: session 2/36's run printed the first session's ID). The PATHS were always
# correct (passed explicitly); only the label was wrong -- but a wrong provenance label in
# every output file is exactly the kind of thing that bites later. Now derived from the
# dataset path's own basename (the session directory name, in both the Oscar and local
# layouts), falling back to the checkpoint's parent directory, then the old default.
if args.session_id is not None:
    SESSION_ID = args.session_id
elif args.dataset_path is not None:
    SESSION_ID = os.path.basename(os.path.normpath(args.dataset_path))
elif args.checkpoint_path is not None:
    SESSION_ID = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint_path)))
else:
    SESSION_ID = "indy_20160407_02"  # this script's original default
CHECKPOINT_DIR   = f"./checkpoints/bmi/mua/indy/{SESSION_ID}"  # trained weights -- stays
# pointed at the original windowed-training checkpoint regardless of which TEST data
# below gets evaluated against it; the model itself wasn't retrained for mua_combined
CHECKPOINT_PATH  = args.checkpoint_path or f"{CHECKPOINT_DIR}/best_model_weights.pth"
SNN_DATASET_PATH = args.dataset_path or f"./datasets/bmi/mua/{SESSION_ID}"  # the
# FULL session's test data as one continuous stream, NOT mua's short 65-timestep/256ms
# windowed trials -- everything below (test_dataset via create_dataloaders, AND the raw
# .pkl trial files run_speck_model()/run_specksim_model() read directly) draws from this
# SAME path, so switching it here is the one change that affects every implementation
# consistently.

active_impls = [m for m in _MODEL_ORDER if m in args.models]
if not active_impls:
    raise ValueError("--models selected zero implementations -- nothing to run")
RUN_ON_SPECK_HARDWARE = "speck" in active_impls
NEEDS_DYNAPCNN = any(m in active_impls for m in ("discretized", "specksim", "speck"))
print(f"Running: {active_impls}")


def unscale_velocity(v_scaled, lo, hi, margin):
    """Inverse of the dataloader's forward scaling. Deliberately NOT
    imported from train_bmi.py -- that module pulls in heavy,
    training-only dependencies (wandb, utils.EarlyStopping/
    save_checkpoint/...) an inference script has no business depending
    on transitively for one small, pure formula. test_all_decoders.py's
    predict_snn_trial() makes the same choice, inlining this exact
    formula rather than importing it, for the same reason."""
    return lo + (hi - lo) * (v_scaled - margin) / (1 - 2 * margin)


def progress_checkpoint(label, t, total, t0, interval=None):
    """Prints a periodic progress line (elapsed time, %, throughput, ETA)
    at a fixed cadence through a long per-timestep loop -- IN ADDITION
    to tqdm's own bar in each runner below, not instead of it.

    Added specifically for mua_combined's full-session single-stream
    runs, which can be thousands of timesteps in ONE call (vs. mua's
    short ~65-timestep windows) -- exactly the situation where "is this
    actually still running, or is it the same kind of silent hang we
    had to hard-kill before" becomes a real, recurring question. tqdm's
    bar answers that visually, but only if it's actually visible (not
    scrolled off, not lost to output buffering over a remote
    connection) -- an explicit, timestamped text line at a fixed
    cadence is a second, independent answer to the same question that
    survives being scrolled past or grepped out of a log file later.

    Defined early in this file (right after unscale_velocity, before any
    runner function below) DELIBERATELY -- run_torch_model() gets called
    once for the cross-check (section 6) before the rest of this script
    has finished executing top-to-bottom, so this needs to already exist
    by then, not just before run_torch_model() is DEFINED.

    interval defaults to roughly every 5% of the run, floored at every
    200 timesteps, so a very long run doesn't spam the log and a short
    one still gets a handful of checkpoints.
    """
    if interval is None:
        interval = max(200, total // 20)
    if t == 0 or (t + 1) % interval == 0 or t + 1 == total:
        elapsed = time.perf_counter() - t0
        pct = (t + 1) / total * 100
        rate = (t + 1) / elapsed if elapsed > 0 else 0
        eta_s = (total - (t + 1)) / rate if rate > 0 else float('nan')
        print(f"  [{label}] {t+1}/{total} ({pct:5.1f}%) -- elapsed {elapsed:7.1f}s, "
              f"{rate:6.1f} steps/s, ETA {eta_s:7.1f}s")

# ---------------------------------------------------------------------------
# Config -- SESSION_ID/CHECKPOINT_PATH/SNN_DATASET_PATH now set above, right
# after argument parsing (see --session-id/--checkpoint-path/--dataset-path)
# ---------------------------------------------------------------------------

BATCH_SIZE      = 1    # inference only -- any value works, since every trial
                        # independently resets regardless of batch size (windowed mode)
N_EVAL_SAMPLES  = 1   # trials sampled for the per-sample snn_seq runner; None = all

PLOT_PREDICTIONS = not NO_PLOTS   # (was hard-coded True; now off under --no-plots) -- render an animation of reconstructed hand position: true vs. decoded
PLOT_MAX_SAMPLES = 16      # only animate the first N of the sampled trials (these are slow to render)
PLOT_MAX_TIMESTEPS = 1000  # cap FRAMES PER ANIMATION -- matplotlib's FuncAnimation with
# writer="pillow" (used throughout this file's plot_sample()/plot_comparison()/
# plot_crosshair_comparison()) buffers EVERY rendered frame in memory before writing the
# GIF out; it does NOT stream frames to disk incrementally. For mua's short ~65-timestep
# windowed trials that's trivial. For mua_combined's full-session single-stream trials
# (tens of thousands of timesteps), it isn't -- confirmed in practice: a real run's
# Speck hardware loop completed 100% cleanly, RMSE/latency printed successfully, then a
# bare "Killed" (no Python traceback -- SIGKILL can't be caught, meaning the Linux OOM
# killer, not an exception) at exactly this next step. Only the FIRST PLOT_MAX_TIMESTEPS
# STRIDED frames of any trial longer than this actually get ANIMATED (see PLOT_STRIDE
# just below for what "strided" means here) -- a full 20,475-frame GIF wouldn't be
# practically watchable anyway (34 minutes at fps=10). Static plots
# (plot_velocity_over_time(), a fig.savefig() PNG, not a FuncAnimation) are NOT capped by
# this -- they have none of the memory risk and showing the full trial there is strictly
# more useful. Set to None to disable the cap entirely (only safe for mua's short trials).
PLOT_STRIDE = 16  # subsample DISPLAYED frames every Nth timestep -- trades animation
# TEMPORAL RESOLUTION for COVERAGE: with PLOT_MAX_TIMESTEPS still capping the animation's
# memory footprint, striding lets those same ~1000 frames span PLOT_STRIDE times more of
# the real session (1000 frames * stride 16 = 16,000 real timesteps covered, instead of
# just the first 1000). CORRECTNESS NOTE, not just a style choice: position is always
# reconstructed from the FULL, UN-strided velocity sequence first (see
# make_position_frames()/reconstruct_path()'s own fixed STEP_TIME-per-sample
# integration) -- striding only subsamples the RESULTING position array afterward, at
# each plotting call site below. Subsampling the raw VELOCITY before integrating would
# silently treat each kept sample as only STEP_TIME apart rather than
# PLOT_STRIDE*STEP_TIME apart, distorting the reconstructed trajectory's actual
# speed/distance -- not just lowering its resolution, genuinely wrong. Set to 1 to
# disable striding (every timestep, up to PLOT_MAX_TIMESTEPS) -- more appropriate for
# mua's short ~65-timestep trials, where PLOT_MAX_TIMESTEPS alone never even triggers
# and striding would just throw away detail for no memory benefit.


def select_plot_indices(T, stride=PLOT_STRIDE, max_frames=PLOT_MAX_TIMESTEPS):
    """Returns the timestep indices to actually ANIMATE for a trial of
    length T -- every `stride`-th index, capped to `max_frames` entries
    (None = uncapped). Apply this to ALREADY-RECONSTRUCTED position (or
    otherwise elementwise-computed) arrays, never to raw velocity before
    integration -- see PLOT_STRIDE's own comment for why that ordering
    matters."""
    indices = list(range(0, T, stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices
PLOT_IMG_SIZE    = 32     # square canvas the reconstructed (x, y) path is scaled onto for display
PLOT_SAVE_DIR    = args.output_dir or "BMI_GIFS"   # directory for .gif/metrics.json
# output; None = inline display only. --output-dir should be passed explicitly when
# running multiple sessions in a loop, so each session's outputs land in their own
# directory rather than overwriting the previous session's.


def display(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------------
# 1. Load checkpoint -- single call, no manual weight-copy loop. See
#    module docstring point 1 for exactly what this replaces and why.
# ---------------------------------------------------------------------------
model, checkpoint, velocity_scale = load_snn_model(CHECKPOINT_PATH, args.experiment)
model.eval()

if args.override_tau_syn is not None:
    # Applied to `model` itself, HERE, before snn_seq (torch)/snn_disc
    # (discretized, rebuilt FROM snn_seq's own layers)/dynapcnn_net (speck/
    # specksim, converted FROM model directly) all get derived from it below
    # -- so every implementation sees the SAME overridden tau_syn, from one
    # shared source, rather than only some of them changing and the
    # comparison drifting apart in a new way. hasattr-based, not isinstance
    # -- catches any layer type carrying a real tau_syn attribute (matches
    # the same duck-typed style already used elsewhere in this file, e.g.
    # run_torch_model()'s own hasattr(m, "spike_threshold") hook check)
    # rather than hardcoding a specific neuron class.
    #
    # BUG FIX: a plain `m.tau_syn = args.override_tau_syn` crashes --
    # confirmed directly against a real checkpoint: "TypeError: cannot
    # assign 'float' as parameter 'tau_syn' (torch.nn.Parameter or None
    # expected)". sinabs registers tau_syn as a real nn.Parameter on these
    # layers (not a plain hyperparameter attribute the way this was
    # originally assumed), so nn.Module's own __setattr__ intercepts the
    # assignment and refuses a bare float. Fixed by modifying the
    # PARAMETER'S OWN .data in place instead of replacing the attribute
    # entirely -- keeps it the same Parameter object (same requires_grad,
    # same identity for anything downstream that already holds a
    # reference to it), just with new numeric content. torch.no_grad()
    # here is belt-and-suspenders, not strictly required for a .data
    # write specifically (.data already bypasses autograd by definition),
    # but makes the intent explicit and costs nothing.
    #
    # The error message's own "...or None expected" confirms tau_syn CAN
    # legitimately be None for a layer that never had synaptic dynamics
    # enabled at all -- there's no existing Parameter there to write
    # .data into, and constructing a brand-new one from scratch would be
    # a different, riskier kind of change (turning on a dynamic that
    # wasn't there, not adjusting one that already was) than what this
    # flag is actually for. Skipped, not silently ignored -- counted and
    # reported separately below so a checkpoint where this turns out to
    # matter doesn't pass through quietly.
    n_overridden = 0
    n_none_skipped = 0
    with torch.no_grad():
        for m in model.modules():
            if not hasattr(m, "tau_syn"):
                continue
            if m.tau_syn is None:
                n_none_skipped += 1
                continue
            m.tau_syn.data.fill_(args.override_tau_syn)
            n_overridden += 1
    print(f"\n{'='*72}\n  DIAGNOSTIC OVERRIDE ACTIVE: tau_syn forced to {args.override_tau_syn} "
          f"on {n_overridden} layer(s) (checkpoint's own trained value ignored). "
          f"Accuracy numbers below are NOT representative of this checkpoint's real, "
          f"trained performance -- this run is testing whether torch/discretized/speck "
          f"move CLOSER together under matched, non-lingering synaptic dynamics, not "
          f"reporting real accuracy.\n{'='*72}")
    if n_none_skipped > 0:
        print(f"  NOTE: {n_none_skipped} layer(s) had tau_syn=None (no synaptic dynamics "
              f"enabled at all for that layer) -- left as None, not overridden. If this "
              f"checkpoint was expected to use tau_syn on every layer, this is worth "
              f"a second look.")
    if n_overridden == 0 and n_none_skipped == 0:
        print("  WARNING: zero layers had a tau_syn attribute to override -- this checkpoint's "
              "own architecture may not use tau_syn at all, or the attribute name has "
              "changed. --override-tau-syn had NO EFFECT this run.")
V_LO, V_HI, V_MARGIN = velocity_scale

train_args = checkpoint.get("args", {})
print(f"Session          : {SESSION_ID}")
print(f"Neuron type      : {train_args.get('neuron_type')}")
print(f"Hidden dims      : {train_args.get('hidden_dims') or [512, 256, 128]}")
print(f"Tau mem          : {train_args.get('tau_mem')}")
print(f"Training mode    : {train_args.get('training_mode', 'windowed')}")
print(f"use_iaf_squeeze  : {train_args.get('use_iaf_squeeze')}")
print(f"Temporal decay   : {model.temporal_decay:.4f} (trained value, correctly restored)")
print(f"Temporal stages  : {model.temporal_decay_stages} (EMA cascade depth; torch/specksim/speck decode through the same number of stages)")

# --- Checkpoint provenance (read-only) ---------------------------------------------------
# Answers, in every session's own log, the questions that otherwise need a separate script:
# which tau_syn does the model this run evaluates ACTUALLY have, did the checkpoint carry
# tau_syn values at all, and what did TRAINING itself report. Added after a real run showed
# model() move 41.81 -> 47.85 and torch 44.74 -> 60.04 while the chip stayed at 45.04 -> 45.11
# (identical weights/thresholds both times) when the loader's tau_syn changed None -> 1.0:
# nothing in the log said which value was in effect, so it had to be inferred from the
# IAFSqueeze repr. best_loss is train_bmi.py's own physical-units test RMSE (sqrt of MSE on
# unscaled velocity), so it is directly comparable to the RMSE numbers printed below -- though
# it comes from the training-time test loader (windowed trials, state reset per trial), not
# this script's single continuous stream, so expect similar, not identical.
def _tau_value(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach()
        return float(t.item()) if t.numel() == 1 else f"tensor{tuple(t.shape)}"
    return float(t)

_constructed_tau_syn = [_tau_value(getattr(_m, "tau_syn", None))
                        for _m in model.modules() if hasattr(_m, "spike_threshold")]
_state_dict_tau_keys = [_k for _k in checkpoint.get("model_state_dict", {}) if _k.endswith(".tau_syn")]
try:
    _ckpt_best_loss = float(checkpoint.get("best_loss")) if checkpoint.get("best_loss") is not None else None
except (TypeError, ValueError):
    _ckpt_best_loss = None
CHECKPOINT_PROVENANCE = {
    "train_args_tau_syn": train_args.get("tau_syn"),
    "n_tau_syn_keys_in_state_dict": len(_state_dict_tau_keys),
    "constructed_tau_syn_per_layer": _constructed_tau_syn,
    "temporal_decay_stages": model.temporal_decay_stages,
    "checkpoint_best_loss": _ckpt_best_loss,
}
print(f"tau_syn          : model built with {_constructed_tau_syn} | checkpoint train_args says "
      f"{train_args.get('tau_syn')} | {len(_state_dict_tau_keys)} tau_syn key(s) in its state_dict")
print(f"Training-time RMSE (checkpoint best_loss, physical units, windowed test loader): "
      f"{_ckpt_best_loss if _ckpt_best_loss is None else round(_ckpt_best_loss, 4)}")

# The Speck2f chip has no synaptic-current stage: DynapcnnNetwork's weights/thresholds and the
# chip's output are identical whatever tau_syn is set to (verified on a real session). So any
# torch model with tau_syn != None is a DIFFERENT network from the one the chip runs -- and, for
# checkpoints from train_bmi_no_tau_syn.py (which builds with tau_syn=None whatever --tau-syn
# says), a different one from the one that was trained. Not an error, because
# --override-tau-syn deliberately sets it for diagnostics, but never silent: a real run
# (indy_20160407_02) scored torch 60.04 vs a faithful 41.81 purely from this, with nothing in the
# log saying so. Recorded in speck_results.json as tau_syn_free so stale/invalid sessions in a
# sweep are identifiable afterwards.
CHECKPOINT_PROVENANCE["tau_syn_free"] = all(_t is None for _t in _constructed_tau_syn)
if not CHECKPOINT_PROVENANCE["tau_syn_free"]:
    print("\n" + "!" * 72)
    print(f"  WARNING: the torch model has tau_syn={_constructed_tau_syn}, which the Speck2f chip cannot")
    print("  implement. Torch-vs-speck differences in this run are NOT like-for-like: the chip runs a")
    print("  tau_syn-free network. If this checkpoint came from train_bmi_no_tau_syn.py it was also")
    print("  TRAINED without tau_syn, so the loader (snn_inference_utils.py) should build with None.")
    print("  (Expected only if you passed --override-tau-syn on purpose.)")
    print("!" * 72 + "\n")

if not train_args.get("use_iaf_squeeze"):
    raise RuntimeError(
        "This checkpoint was trained with use_iaf_squeeze=False. The dynapcnn-prep path "
        "below (snn_seq) specifically requires Squeeze-variant layers -- see module "
        "docstring point 3 for why bypassing _SinabsNeuronLayer's wrapper is only valid in "
        "that case. This isn't just a script limitation: Speck deployment itself requires "
        "this layer convention, so a non-Squeeze checkpoint would need retraining with "
        "--use-iaf-squeeze before this path is meaningful at all, not a workaround here.")

# ---------------------------------------------------------------------------
# 2. Dataloader
# ---------------------------------------------------------------------------
from datasets.dataset import create_dataloaders

# experiment/subject passed EXPLICITLY. BUG FIX: the new dataset.py auto-derives them from
# data_path ('.../{experiment}/{subject}/...'), which works on the Oscar layout
# (snn_datasets/bmi/indy/mua_8_group/{session}) but NOT this machine's asymmetric one
# (datasets/bmi/mua/{session}, no subject directory level -- see
# run_speck_inference_sweep.sh's own header): the component after 'bmi' is 'mua', not a
# subject, so derivation raised ValueError. Reproduced directly against the real dataset.py
# and this exact layout before fixing. args.subject=None still falls back to path
# derivation, so layouts where that works behave exactly as before.
try:
    train_loader, test_loader = create_dataloaders(
        data_path=SNN_DATASET_PATH, batch_size=BATCH_SIZE, num_workers=0,
        shuffle_train=False, small=False,
        experiment=args.experiment, subject=args.subject,
    )
except ValueError as exc:
    if args.subject is None and "Could not derive (experiment, subject)" in str(exc):
        raise ValueError(
            f"{exc}\n\n  -> From this script's CLI: pass --subject <subject> "
            f"(e.g. --subject indy) so dataset.py doesn't have to guess it from the path.") from exc
    raise
print(f"Test dataset: {len(test_loader)} batches of up to {BATCH_SIZE}")

# --- Velocity-scaling consistency check ---
# Two INDEPENDENT sources of velocity bounds meet in this script: the dataset scales TARGETS
# with velocity_scalers.json's per-subject (v_lo, v_hi, margin), while V_LO/V_HI/V_MARGIN
# above (used by every unscale_velocity() call below) come from the CHECKPOINT's own stored
# train_args -- with a silent fallback to legacy, Indy-only constants if the checkpoint
# predates those args (see snn_inference_utils.py's _FALLBACK_VELOCITY_*; test_all_decoders.py
# has the identical structure). Nothing else in this pipeline verifies the two agree, so an old
# checkpoint evaluated on a non-Indy subject would just quietly report distorted physical-unit
# numbers. Checked once here and recorded in speck_results.json so a mismatch can't hide in a
# 36-session sweep's scrollback.
_ds = test_loader.dataset
_checkpoint_scale = (V_LO, V_HI, V_MARGIN)
_dataset_scale = (getattr(_ds, "v_lo", None), getattr(_ds, "v_hi", None),
                  getattr(_ds, "v_margin", None))
VELOCITY_SCALE_CHECKPOINT = [float(v) for v in _checkpoint_scale]
VELOCITY_SCALE_DATASET = None if None in _dataset_scale else [float(v) for v in _dataset_scale]
if VELOCITY_SCALE_DATASET is None:
    VELOCITY_SCALING_CONSISTENT = None
    print("[scaling check skipped] the loaded dataset exposes no v_lo/v_hi/v_margin "
          "(an older dataset.py?) -- cannot compare against the checkpoint's own bounds.")
elif np.allclose(_checkpoint_scale, _dataset_scale):
    VELOCITY_SCALING_CONSISTENT = True
    print(f"Velocity scaling : checkpoint == dataset "
          f"(lo={V_LO:.2f}, hi={V_HI:.2f}, margin={V_MARGIN}) -- consistent")
else:
    VELOCITY_SCALING_CONSISTENT = False
    print("\n" + "!" * 72)
    print("  WARNING: velocity-scaling MISMATCH between checkpoint and dataset")
    print(f"    checkpoint (used to unscale every prediction/target below): "
          f"lo={V_LO}, hi={V_HI}, margin={V_MARGIN}")
    print(f"    dataset    (used to scale the targets this model is scored against): "
          f"lo={_dataset_scale[0]}, hi={_dataset_scale[1]}, margin={_dataset_scale[2]}")
    print("    Physical-unit RMSE below is DISTORTED by this (CC is not: Pearson r is "
          "invariant to an affine rescale of either side). Likely causes: this checkpoint "
          "predates velocity_lo/hi/margin being saved in train_args (so the legacy Indy "
          "constants were silently substituted), or it was trained against a different "
          "velocity_scalers.json than the one this dataset.py is reading.")
    print("!" * 72 + "\n")

# ---------------------------------------------------------------------------
# 3. Full-batch eval via SNN_Speck.forward() -- physical velocity units,
#    matching train_bmi.py's/test_all_decoders.py's own convention (see
#    module docstring point 4).
# ---------------------------------------------------------------------------
criterion = torch.nn.MSELoss()
all_losses = []

with torch.no_grad():
    for labels, inputs, targets in test_loader:
        inputs = inputs.transpose(0, 1)   # (N,T,C) -> (T,N,C)
        targets = targets.transpose(0, 1)  # (N,T,2) -> (T,N,2)
        predictions, _, _ = model(inputs)  # windowed -> forward() always resets per call now
        # (model_bmi.py's own forward() no longer accepts a reset_state argument at all --
        # removed along with the dead "continuous training" machinery it used to support;
        # it always resets unconditionally now, which is exactly the behavior reset_state=True
        # used to request explicitly, so this call is behaviorally identical to before)
        pred_phys = unscale_velocity(predictions, V_LO, V_HI, V_MARGIN)
        target_phys = unscale_velocity(targets, V_LO, V_HI, V_MARGIN)
        rmse = torch.sqrt(criterion(pred_phys, target_phys))  # RMSE, not raw MSE -- matches
        # train_bmi.py's/test_all_decoders.py's own reporting convention throughout this
        # project. Raw MSE looks alarmingly large purely because it's squared, not because
        # anything is actually wrong -- e.g. MSE~2400 is RMSE~49, well within this project's
        # normal range.
        all_losses.append(rmse.item())

print(f"\n[model()] Test RMSE (physical velocity units): "
      f"{np.mean(all_losses):.4f} +/- {np.std(all_losses):.4f}")

# ---------------------------------------------------------------------------
# 4. Build a plain nn.Sequential for dynapcnn/hardware-deployment-style use.
#    See module docstring point 3 for the use_iaf_squeeze requirement this
#    depends on (already asserted above).
# ---------------------------------------------------------------------------
seq_layers = []
for m in model.layers:
    if isinstance(m, (_SinabsNeuronLayer, _SJNeuronLayer)):
        seq_layers.append(m.neuron)
    else:
        seq_layers.append(m)

snn_seq = nn.Sequential(*seq_layers)
print("\nsnn_seq (dynapcnn-prep):")
print(snn_seq)

# ---------------------------------------------------------------------------
# 5. Per-sample runner with the correct causal EMA decode
# ---------------------------------------------------------------------------
N_BINS = model.n_bins
POSITIONS = model.positions.clone()
NEUTRAL_SCALED_PRED = model.neutral_scaled_pred.clone()
TEMPORAL_DECAY = model.temporal_decay
TEMPORAL_DECAY_STAGES = model.temporal_decay_stages


def decode_from_ema(ema: torch.Tensor) -> torch.Tensor:
    """(1, N_BINS*2) EMA-smoothed state -> (1, 2) (x, y) in SCALED space --
    reimplements SNN_Speck.decode_output() EXACTLY, including the
    zero-spike neutral_scaled_pred fallback (see module docstring point
    2). Required outside the model because Speck/DynapcnnNetwork can
    only execute the raw neuron layers themselves, never this project's
    Python-side decode logic -- that logic has to be reimplemented
    somewhere for a hardware-facing runner, but every value it depends
    on is pulled from the loaded model itself, not re-declared, so it
    can't silently drift the way the previous version of this script did.
    """
    pos = POSITIONS.to(ema.device)
    x_acc = ema[:, :N_BINS]
    y_acc = ema[:, N_BINS:N_BINS * 2]
    x_raw_total = x_acc.sum(dim=1)
    y_raw_total = y_acc.sum(dim=1)
    x_has_signal = x_raw_total > 0
    y_has_signal = y_raw_total > 0
    x_total = x_raw_total.clamp(min=1)
    y_total = y_raw_total.clamp(min=1)
    pred_x_raw = (x_acc * pos).sum(dim=1) / x_total
    pred_y_raw = (y_acc * pos).sum(dim=1) / y_total
    neutral = NEUTRAL_SCALED_PRED.to(ema.device)
    pred_x = torch.where(x_has_signal, pred_x_raw, neutral)
    pred_y = torch.where(y_has_signal, pred_y_raw, neutral)
    return torch.stack([pred_x, pred_y], dim=-1)


def ema_cascade_update(ema_stages: list, x: torch.Tensor, decay: float) -> torch.Tensor:
    """One timestep of SNN_Speck.forward()'s temporal EMA CASCADE: stage 0 is fed
    this timestep's raw output layer activity, each later stage is fed the PREVIOUS
    stage's own EMA output, and decoding reads the LAST stage. Updates `ema_stages`
    (a list of (1, 2*N_BINS) tensors, one per stage) in place and returns the last
    stage, i.e. the tensor to pass to decode_from_ema().

    BUG FIX: every EMA in this script (run_torch_model, and the specksim and speck
    loops) used to be ONE stage -- `ema = decay * ema + x` -- while the model itself
    cascades `temporal_decay_stages` of them (model_bmi.py's forward(): 'stage_input =
    x; for i in range(self.temporal_decay_stages): ema[i] = decay * ema[i] +
    stage_input; stage_input = ema[i]', decoding from ema[-1]). This script never read
    model.temporal_decay_stages at all. With stages=1 the two are identical -- which is
    why the model()-vs-snn_seq() cross-check used to pass -- but for any checkpoint
    trained with stages >= 2 (this project's settled config uses 2) torch/specksim/
    speck were all decoded through a DIFFERENT, less-smoothed temporal filter than the
    one the checkpoint was trained with and that model() itself applies. Confirmed
    from a real run: the cross-check reported max |model() - snn_seq()| = 0.135 in
    scaled velocity units (WARNING printed), and model() scored RMSE 41.81 vs 44.74
    for snn_seq on the identical trial, ground truth and scoring window. The chip
    only replaces the spiking layers, so this host-side filter is shared by all three
    implementations -- it is why torch and speck tracked each other while both
    differed from model(). With stages=1 this reduces exactly to the old behaviour.
    """
    stage_input = x
    for i in range(len(ema_stages)):
        ema_stages[i] = decay * ema_stages[i] + stage_input
        stage_input = ema_stages[i]
    return ema_stages[-1]


def run_torch_model(seq_model: nn.Sequential, input_spikes: torch.Tensor,
                     decay: float = TEMPORAL_DECAY):
    """input_spikes: (T, C) -- one trial's raw spike raster, timestep-major
    (matches SNN_Speck.forward()'s own x_total convention, NOT the SNN
    .pkl files' own (C, T) on-disk storage -- transpose before calling).
    Returns (predictions (T, 2) in SCALED space, per-layer spike counts,
    latency_s, energy_j_low, energy_j_high, energy_method).

    latency_s is measured over ONLY the per-timestep decode loop below --
    NOT sinabs.reset_states()/hook registration/model.eval() before it, or
    hook-removal/reset after. Requested directly (for Speck's own
    v_mem-reset overhead, but applied uniformly here too): setup isn't
    part of "decoding," so it shouldn't count toward a number meant to
    answer "how long does decoding take." For torch this setup is
    genuinely tiny (microseconds) -- unlike Speck's, where v_mem reset +
    settling is a real, multi-second cost -- but scoping every
    implementation's latency to the identical kind of window keeps them
    all answering the exact same question, not just Speck's.

    energy_j_low/energy_j_high: REPLACES the old EnergyMeter-based
    host-CPU approximation (rapl/proxy_psutil) with op_energy_estimate.py's
    MAC/ACC-based estimate instead -- the same methodology test_all_decoders.py
    already uses for KF/WF/MLP/LSTM/QRNN and for this SNN's own comparison
    there, so torch/discretized numbers here are now directly comparable
    to that report rather than measured by a fundamentally different
    method. Genuinely a RANGE (SRAM-only vs. DRAM-only memory-access
    assumption, ~128x apart) by the estimator's own design -- reporting a
    single collapsed number would imply a precision the method doesn't
    actually have (see op_energy_estimate.py's own module docstring).
    Computed via a SEPARATE call to the original `model` object (not
    `seq_model`/snn_seq -- that's a manually-unwrapped nn.Sequential with
    no count_ops support of its own), on the same input_spikes trial;
    model() and snn_seq() are already cross-checked elsewhere in this
    script to produce identical predictions, so this separate call's op
    counts are a valid, representative estimate of what seq_model also
    computes.

    Maintains a persistent causal EMA across the whole sequence, exactly
    mirroring SNN_Speck.forward()'s internal temporal_ema update -- one
    new timestep fed in at a time, decoding continuously from
    accumulated state, which is also exactly how this would actually run
    on hardware.
    """
    iaf_layers = [m for m in seq_model.modules() if hasattr(m, "spike_threshold")]
    n_layers = len(iaf_layers)
    spike_counts = {i: [] for i in range(n_layers)}
    frame_counts_log = [0] * n_layers

    def make_hook(idx):
        def hook(module, inp, out):
            frame_counts_log[idx] = int(out.sum().item())
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(iaf_layers)]

    import sinabs
    sinabs.reset_states(seq_model)
    seq_model.eval()

    predictions = []
    ema_stages = [torch.zeros(1, N_BINS * 2) for _ in range(TEMPORAL_DECAY_STAGES)]
    T = input_spikes.shape[0]

    with EnergyMeter() as em:  # latency_s only now -- energy_j/energy_method no
        # longer used from here, see docstring above
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in tqdm(range(T), desc="Torch (snn_seq)"):
                x = seq_model(input_spikes[t].unsqueeze(0))  # (1, C) -> ... -> (1, 2*N_BINS)
                ema = ema_cascade_update(ema_stages, x, decay)
                pred = decode_from_ema(ema)  # (1, 2)
                predictions.append(pred.squeeze(0))
                for i in range(n_layers):
                    spike_counts[i].append(frame_counts_log[i])
                progress_checkpoint("torch", t, T, t0)

    for h in handles:
        h.remove()
    sinabs.reset_states(seq_model)

    # MAC/ACC-based energy estimate -- separate call to the ORIGINAL model
    # object (has count_ops support; seq_model/snn_seq does not), same
    # trial, count_ops=True.
    with torch.no_grad():
        _, _, _, op_counts = model(input_spikes.unsqueeze(1), count_ops=True)  # (T,C) -> (T,1,C)
    op_estimate = finalize_snn_ops(op_counts['mac'], op_counts['acc'],
                                    op_counts['elementwise'], n_samples=T)

    return (torch.stack(predictions, dim=0), spike_counts, em.latency_s,
            op_estimate['energy_total_j_low'], op_estimate['energy_total_j_high'],
            "mac_acc_estimate")


# ---------------------------------------------------------------------------
# 6. Cross-check: model() vs. snn_seq() on the SAME real trial. The old
#    script built both paths but never directly confirmed they agree --
#    worth catching a disagreement HERE, before dynapcnn/hardware is even
#    in the picture (see module docstring point 5).
# ---------------------------------------------------------------------------
_check_batch = next(iter(test_loader))
_, _check_inputs, _ = _check_batch
_check_trial = _check_inputs[0].float()  # (T, C) -- one real trial, timestep-major already

with torch.no_grad():
    _model_pred, _, _ = model(_check_trial.unsqueeze(1))  # (T,1,C) -> (T,1,2) -- see the other
    # call site's own comment: forward() always resets now, no reset_state argument needed/accepted
_model_pred = _model_pred.squeeze(1)  # (T, 2)

_seq_pred, _, _, _, _, _ = run_torch_model(snn_seq, _check_trial)  # (T, 2)

_max_diff = (_model_pred - _seq_pred).abs().max().item()
print(f"\n[cross-check] max |model() - snn_seq()| over one real trial: {_max_diff:.6f}")
if _max_diff > 1e-4:
    print("  WARNING: model() and snn_seq() disagree beyond floating-point tolerance -- "
          "the dynapcnn-prep path is not yet a faithful reproduction of the real model. "
          "Do not proceed to DynapcnnNetwork conversion until this is resolved.")
else:
    print("  OK: snn_seq() faithfully reproduces model()'s own predictions on real data.")

# ---------------------------------------------------------------------------
# 7. Visualization: reconstructed hand position, true vs. decoded (see
#    module docstring point 6 for why this replaces the old fixed-canvas
#    2D image tracking, and why that's a stated assumption, not an
#    obvious translation).
# ---------------------------------------------------------------------------
STEP_TIME = 0.004  # this project's fixed native sampling interval


def make_position_frames(velocity_scaled_pred: np.ndarray, velocity_scaled_true: np.ndarray,
                          img_size: int = PLOT_IMG_SIZE):
    """velocity_scaled_*: (T, 2) in SCALED [0,1]-ish space, as returned by
    decode_from_ema()/model(). Unscales to physical velocity, reconstructs
    position via integration (anchored at (0,0) -- there's no true
    absolute position in this dataset, only velocity, so this shows
    RELATIVE displacement over the trial, not absolute hand position),
    then rescales both traces into a shared [0, img_size) pixel frame
    for display.
    """
    vel_pred_phys = unscale_velocity(velocity_scaled_pred, V_LO, V_HI, V_MARGIN)
    vel_true_phys = unscale_velocity(velocity_scaled_true, V_LO, V_HI, V_MARGIN)

    anchor = np.zeros(2)
    pos_pred = reconstruct_path(anchor, vel_pred_phys, STEP_TIME)
    pos_true = reconstruct_path(anchor, vel_true_phys, STEP_TIME)

    # Shared scale across BOTH traces, so their relative sizes stay
    # meaningful in the shared frame rather than each being independently
    # normalized to fill the canvas.
    all_pos = np.concatenate([pos_pred, pos_true], axis=0)
    lo, hi = all_pos.min(axis=0), all_pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def to_pixels(pos):
        return ((pos - lo) / span * (img_size - 1)).astype(int)

    return to_pixels(pos_pred), to_pixels(pos_true)


def plot_sample(batch_idx: int, sample_idx: int, pos_pred_px: np.ndarray, pos_true_px: np.ndarray,
                 img_size: int = PLOT_IMG_SIZE, save_dir: str = PLOT_SAVE_DIR):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_title(f"Sample {batch_idx}_{sample_idx}: true (black) vs. decoded (blue)")

    true_line, = ax.plot([], [], color='black', linewidth=1.2, label='true')
    pred_line, = ax.plot([], [], color='royalblue', linewidth=1.2, linestyle='--', label='decoded')
    true_dot, = ax.plot([], [], marker='o', color='black', markersize=6)
    pred_dot, = ax.plot([], [], marker='o', color='royalblue', markersize=6)
    ax.legend(loc='upper right', fontsize=8)

    def update(t):
        true_line.set_data(pos_true_px[:t + 1, 0], pos_true_px[:t + 1, 1])
        pred_line.set_data(pos_pred_px[:t + 1, 0], pos_pred_px[:t + 1, 1])
        true_dot.set_data([pos_true_px[t, 0]], [pos_true_px[t, 1]])
        pred_dot.set_data([pos_pred_px[t, 0]], [pos_pred_px[t, 1]])
        return true_line, pred_line, true_dot, pred_dot

    ani = FuncAnimation(fig, update, frames=pos_true_px.shape[0], interval=100)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ani.save(f"{save_dir}/prediction_{batch_idx}_{sample_idx}.gif", writer="pillow", fps=10)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Evaluate the per-sample (snn_seq) runner against the same test set
#
# Samples EVENLY SPACED across the WHOLE test set (not just the first
# N_EVAL_SAMPLES sequentially) -- matching check_snn_predictions.py's own
# --sample spread convention. A run of N sequential trials from the start
# of the test set can look very different from the test set as a whole if
# performance isn't uniform over the session (electrode drift, non-
# stationarity -- a real, common phenomenon in this data, not a bug) --
# spreading the sample across the full range is what actually answers
# "does this look right across the whole range of trial activity."
# ---------------------------------------------------------------------------
def select_evenly_spaced_indices(n_total, n_requested):
    if n_requested is None or n_requested >= n_total:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, n_requested).astype(int).tolist()))


all_losses_seq = []
plotted_samples = 0

test_dataset = test_loader.dataset
print(f"\ntest_dataset has {len(test_dataset)} trial(s) at {SNN_DATASET_PATH!r}")
if len(test_dataset) == 1:
    print("  -> exactly one trial, i.e. the full session as a single continuous "
          "stream -- this is what 'single stream' processing below assumes.")
else:
    print(f"  -> WARNING: {len(test_dataset)} SEPARATE trials found, not one -- "
          f"if you intended mua_combined to be a single continuous stream, this "
          f"directory doesn't have that structure. The eval loop below (N_COMPARE) "
          f"will still only process ONE of these ({len(test_dataset)} trials get "
          f"treated as {len(test_dataset)} INDEPENDENT, separately-reset windows, "
          f"not concatenated into one continuous pass) -- say so explicitly if you "
          f"actually want them concatenated; that's a different, real code change "
          f"(deciding where resets happen across the boundary), not a config tweak.")

eval_indices = select_evenly_spaced_indices(len(test_dataset), N_EVAL_SAMPLES)
print(f"\nEvaluating {len(eval_indices)} trials, evenly spaced across all "
      f"{len(test_dataset)} test trials: indices {eval_indices}")

for trial_idx in tqdm(eval_indices, desc="Evaluating (snn_seq)"):
    _, input_trial, target_trial = test_dataset[trial_idx]
    input_trial = input_trial.float()    # (T, C)
    target_trial = target_trial.float()  # (T, 2), SCALED space

    pred_scaled, sc_torch, _, _, _, _ = run_torch_model(snn_seq, input_trial)

    pred_phys = unscale_velocity(pred_scaled, V_LO, V_HI, V_MARGIN)
    target_phys = unscale_velocity(target_trial, V_LO, V_HI, V_MARGIN)
    rmse = torch.sqrt(criterion(pred_phys, target_phys))
    all_losses_seq.append(rmse.item())

    if PLOT_PREDICTIONS and plotted_samples < PLOT_MAX_SAMPLES:
        T_trial = pred_scaled.shape[0]
        plot_idx = select_plot_indices(T_trial)
        if len(plot_idx) < T_trial:
            print(f"  [plot] trial {trial_idx} has {T_trial} timesteps -- animating "
                  f"{len(plot_idx)} frames (every {PLOT_STRIDE}th), covering up to timestep "
                  f"{plot_idx[-1]} ({(plot_idx[-1] + 1) / T_trial * 100:.1f}% of the trial). "
                  f"Position is reconstructed from the FULL trial first, THEN subsampled for "
                  f"display -- see PLOT_STRIDE's own comment for why that order matters.")
        pos_pred_px, pos_true_px = make_position_frames(
            pred_scaled.numpy(), target_trial.numpy())  # FULL trial -- reconstruction is a
        # cheap vectorized cumulative sum regardless of length; only ANIMATING every frame
        # is the expensive/memory-risky part, handled by the indexing right below
        pos_pred_px, pos_true_px = pos_pred_px[plot_idx], pos_true_px[plot_idx]
        plot_sample(0, trial_idx, pos_pred_px, pos_true_px)
        plotted_samples += 1

print(f"\n[snn_seq] Test RMSE over {len(eval_indices)} trials (physical velocity units): "
      f"{np.mean(all_losses_seq):.4f} +/- {np.std(all_losses_seq):.4f}")

DYNAPCNN_PARAM_COUNT = None  # set below only if NEEDS_DYNAPCNN -- stays None (not
# an error) for a --models torch-only run, where no DynapcnnNetwork ever gets built at all.

if NEEDS_DYNAPCNN:
    # ---------------------------------------------------------------------------
    # 9. DynapcnnNetwork conversion + SpeckSim -- first pass at actual
    #    hardware/simulator deployment. NOTE: neither DynapcnnNetwork nor
    #    SpeckSim (which need samna, a hardware-specific binary dependency)
    #    could be installed or run in the sandbox this was developed in --
    #    everything below is verified against sinabs' own current
    #    documentation (import paths, the Linear->Conv2d(1x1) auto-conversion,
    #    the IAF-only requirement) but NOT run end to end against the real
    #    libraries. Treat this section as a well-researched first draft to
    #    run and debug on your own system, not a tested deliverable the way
    #    everything else in this project has been.
    #
    # WHAT CHANGED vs. the old script's version, and why:
    #
    #   - Weight-norm stripping (the old script's step 6) is removed entirely.
    #     This architecture never uses weight-norm parametrization at all
    #     (plain nn.Linear, no Conv2d, no parametrizations registered) -- that
    #     whole block would be a silent no-op here, not a needed step.
    #
    #   - input_shape is (num_input_channels, 1, 1), not a real (C, H, W)
    #     image. Confirmed via sinabs' own current docs: DynapcnnNetwork
    #     automatically converts nn.Linear layers to an equivalent 1x1
    #     nn.Conv2d internally -- our purely-Linear architecture is
    #     deployable, just with no spatial extent to the "image" it's
    #     nominally processing.
    #
    #   - dvs_input=False, not True. The old script's dvs_input=True made
    #     sense for its actual DVS/vision-camera-style input; this project's
    #     input is MUA spike counts arriving via a different interface
    #     entirely, not the chip's integrated DVS sensor. This is a real,
    #     load-bearing configuration choice, not a renamed default -- worth
    #     confirming this is what you actually want before deploying.
    #
    #   - An explicit assertion that neuron_type == 'iaf' before proceeding.
    #     Specksim/DynapcnnNetwork documentation states only
    #     sinabs.layers.IAF/IAFSqueeze are supported -- LIF is not. This is a
    #     genuine hardware requirement, not a script limitation to work
    #     around: a LIF-trained checkpoint would need retraining as IAF
    #     before this section is meaningful at all.
    #
    #   - The snn_disc rebuild loop only handles nn.Linear and IAFSqueeze --
    #     the old script's nn.Conv2d/sl.SumPool2d/nn.Flatten branches are
    #     removed, since this architecture never contains those layer types.
    #     The "unhandled layer type" fallback is kept, so a future
    #     architecture change that DOES introduce one of those fails loudly
    #     here rather than silently mishandling it.
    #
    #   - The bare `except:` around the dual dynapcnn_net.sequence /
    #     dynapcnn_net.dynapcnn_layers attribute-access fallback (present in
    #     the old script, protecting against a real, documented API change
    #     across sinabs-dynapcnn versions) is narrowed to `except
    #     AttributeError:` specifically -- a bare except also swallows
    #     genuine bugs, not just the intended API-version fallback.
    # ---------------------------------------------------------------------------
    if train_args.get("neuron_type") != "iaf":
        raise RuntimeError(
            f"This checkpoint was trained with neuron_type={train_args.get('neuron_type')!r}. "
            f"Specksim/DynapcnnNetwork documentation states only sinabs.layers.IAF/IAFSqueeze "
            f"are supported for actual chip deployment -- LIF is not. This session would need "
            f"retraining with --neuron-type iaf before this section is meaningful.")

    from sinabs.backend.dynapcnn import DynapcnnNetwork
    from sinabs.backend.dynapcnn.specksim import from_sequential
    import sinabs.layers as sl

    DYNAPCNN_INPUT_SHAPE = (int(model.layers[0].in_features), 1, 1)
    print(f"\nDynapcnnNetwork input_shape: {DYNAPCNN_INPUT_SHAPE} "
          f"(no spatial structure -- this project's input is a flat channel vector, "
          f"not a real image)")

    # DynapcnnNetwork traces the model's graph with a dummy input matching
    # the Conv2d (N, C, H, W) convention -- (1, 96, 1, 1) here. nn.Linear
    # treats the LAST dimension as in_features regardless of how many other
    # dims precede it; for a (1, 96, 1, 1) input the last dim is 1 (width),
    # not 96 (channels), so a bare Linear-first Sequential fails during
    # tracing itself, before any Linear->Conv2d(1x1) conversion even
    # happens. Confirmed directly: reproduced this exact failure
    # ("mat1 and mat2 shapes cannot be multiplied (96x1 and 96x256)") with
    # plain PyTorch, no dynapcnn/samna needed to see it -- and confirmed
    # prepending nn.Flatten() (matching the exact structure of sinabs' own
    # documented DynapcnnNetwork examples, which always have Flatten()
    # immediately before their first Linear) resolves it, while being a
    # verified no-op for the (1, C)-shaped input run_torch_model()'s
    # already-tested cross-check path uses. Built as a SEPARATE Sequential
    # here, not by modifying snn_seq itself, so nothing already verified
    # above is put at risk by this change.
    snn_seq_for_dynapcnn = nn.Sequential(nn.Flatten(), *snn_seq)

    dynapcnn_net = DynapcnnNetwork(
        snn=snn_seq_for_dynapcnn,
        input_shape=DYNAPCNN_INPUT_SHAPE,
        discretize=True,
        dvs_input=False,  # NOT a vision/DVS sensor input -- see module comment above
    )
    print("\ndynapcnn_net:")
    print(dynapcnn_net)

    # --- Diagnostic: does the CONVERTED, chip-bound neuron object even
    # HAVE a tau_syn attribute at all? -- built specifically to answer a
    # real, open question directly rather than guess a third time: with
    # --override-tau-syn active, torch/discretized both clearly reflect
    # the override (RMSE changes dramatically), but speck's own numbers
    # come back numerically identical to an un-overridden run -- strongly
    # suggesting the chip-bound object never receives tau_syn at all,
    # possibly because IAF (the only neuron type DynapcnnNetwork/specksim
    # actually supports for deployment -- LIF is documented as
    # unsupported, see this section's own header comment) may have no
    # hardware-realized synaptic-current stage to even configure. Purely
    # read-only/exploratory -- prints what's actually there rather than
    # assuming an attribute path and risking a third wrong guess. Same
    # dynapcnn_net.sequence / dynapcnn_net.dynapcnn_layers fallback this
    # file already uses just below (disc_weights extraction), for the
    # same real, documented sinabs-dynapcnn API-version reason.
    print("\n--- tau_syn diagnostic: does the chip-bound layer object expose it at all? ---")
    try:
        chip_layers_to_inspect = [(i, layer.spk_layer) for i, layer in enumerate(dynapcnn_net.sequence)
                                   if hasattr(layer, "spk_layer")]
    except AttributeError:
        chip_layers_to_inspect = [(idx, dynapcnn_net.dynapcnn_layers[idx].spk)
                                   for idx in sorted(dynapcnn_net.dynapcnn_layers.keys())]
    for i, spk_layer in chip_layers_to_inspect:
        has_tau_syn = hasattr(spk_layer, "tau_syn")
        tau_syn_value = getattr(spk_layer, "tau_syn", "N/A -- attribute does not exist")
        print(f"  layer {i} ({type(spk_layer).__name__}): hasattr(tau_syn)={has_tau_syn}, "
              f"value={tau_syn_value}")
    if chip_layers_to_inspect:
        first_type_name, first_layer = type(chip_layers_to_inspect[0][1]).__name__, chip_layers_to_inspect[0][1]
        public_attrs = sorted(a for a in dir(first_layer) if not a.startswith("_"))
        print(f"  Full public attribute list of layer 0 ({first_type_name}), for reference: "
              f"{public_attrs}")
    print("--- end tau_syn diagnostic ---\n")

    # Model size of what ACTUALLY gets deployed to the chip -- total
    # parameter count (weights + biases) across dynapcnn_net's own
    # converted layers, not the original pre-conversion model. Same
    # layer.conv_layer attribute access already proven to work against
    # this exact object just below (disc_weights extraction) -- reused
    # here rather than a separate, unverified traversal.
    DYNAPCNN_PARAM_COUNT = 0
    for layer in dynapcnn_net.sequence:
        try:
            DYNAPCNN_PARAM_COUNT += layer.conv_layer.weight.numel()
        except AttributeError:
            pass
        try:
            DYNAPCNN_PARAM_COUNT += layer.conv_layer.bias.numel()
        except AttributeError:
            pass
    print(f"DynapcnnNetwork total parameter count (weights + biases, as actually "
          f"deployed): {DYNAPCNN_PARAM_COUNT:,}")

    # ---------------------------------------------------------------------------
    # 10. Extract discretized weights + thresholds
    # ---------------------------------------------------------------------------
    disc_weights = []
    disc_thresholds = []
    disc_min_v_mem = []   # see the BUG FIX note in the snn_disc copy loop below

    try:
        for layer in dynapcnn_net.sequence:
            try:
                disc_thresholds.append(layer.spk_layer.spike_threshold)
                disc_min_v_mem.append(layer.spk_layer.min_v_mem)
            except AttributeError:
                pass
            try:
                disc_weights.append(layer.conv_layer.weight.data.clone())
            except AttributeError:
                pass
    except AttributeError:
        for idx in sorted(dynapcnn_net.dynapcnn_layers.keys()):
            layer = dynapcnn_net.dynapcnn_layers[idx]
            disc_thresholds.append(layer.spk.spike_threshold)
            disc_min_v_mem.append(layer.spk.min_v_mem)
            disc_weights.append(layer.conv.weight.data.clone())

    print(f"Discretized thresholds : {disc_thresholds}")
    print(f"Discretized min_v_mem  : {disc_min_v_mem}")
    print(f"Discretized weight layers: {len(disc_weights)}")
    assert len(disc_min_v_mem) == len(disc_thresholds), (
        f"got {len(disc_min_v_mem)} min_v_mem values for {len(disc_thresholds)} thresholds -- "
        f"the chip-bound layers must each expose both; refusing to build snn_disc with a "
        f"misaligned or partially-copied membrane floor.")
    assert len(disc_weights) == len(disc_thresholds), (
        f"Expected one weight tensor per threshold (one Linear->Conv2d(1x1) per spiking layer), "
        f"got {len(disc_weights)} weight tensors and {len(disc_thresholds)} thresholds -- "
        f"something about this architecture doesn't match the expected Linear-then-IAF "
        f"alternating pattern. Do not proceed until this is understood.")

    # ---------------------------------------------------------------------------
    # 11. Build snn_disc -- same topology as snn_seq, discretized weights +
    #     thresholds. Only Linear and IAFSqueeze handled (see module comment
    #     above for why the old script's Conv2d/SumPool2d/Flatten branches
    #     were dropped, not just left in unused).
    # ---------------------------------------------------------------------------
    def _rebuild_layer(m):
        if isinstance(m, nn.Linear):
            return nn.Linear(m.in_features, m.out_features, bias=False)
        if isinstance(m, sl.IAFSqueeze):
            # BUG FIX: this constructor call previously omitted tau_syn
            # entirely -- silently falling back to sinabs' own default
            # (None, i.e. NO synaptic-current filtering stage at all),
            # even though model_bmi.py's own NeuronFactory.build() passes
            # tau_syn=self.tau_syn in EVERY neuron-construction branch
            # (confirmed directly), and this project's own settled,
            # deployed config trains with tau_syn=8.0 -- a real, trained
            # part of the network's own dynamics, not an optional extra.
            # discretized was the ONLY one of torch/discretized/specksim/
            # speck missing it: torch uses the original model object
            # directly, and specksim/speck deploy straight from
            # dynapcnn_net, never passing through this rebuild function at
            # all -- so this bug affected discretized's own accuracy
            # specifically, not the others. getattr with a loud warning
            # (not silent None) if snn_seq's own IAFSqueeze layer somehow
            # doesn't expose tau_syn as a readable attribute -- min_v_mem
            # is already read the same way just below (m.min_v_mem,
            # confirmed working), so this is expected to work the same
            # way, but failing loudly rather than silently assuming so if
            # a sinabs version difference ever makes that untrue.
            m_tau_syn = getattr(m, "tau_syn", "MISSING")
            if m_tau_syn == "MISSING":
                print(f"  WARNING: {type(m).__name__} has no readable .tau_syn attribute -- "
                      f"falling back to tau_syn=None (sinabs' own default, NO synaptic "
                      f"filtering) for snn_disc. This is very likely WRONG if the checkpoint "
                      f"was trained with tau_syn set -- flag and investigate before trusting "
                      f"'discretized' results.")
                m_tau_syn = None
            return sl.IAFSqueeze(
                spike_threshold=1.0, min_v_mem=float(m.min_v_mem), num_timesteps=1,
                surrogate_grad_fn=m.surrogate_grad_fn, reset_fn=m.reset_fn, spike_fn=m.spike_fn,
                tau_syn=m_tau_syn,
            )
        raise ValueError(
            f"Unhandled layer type: {type(m)} -- this architecture is expected to contain only "
            f"nn.Linear and sl.IAFSqueeze layers. A different layer type showing up here means "
            f"either the architecture changed (and this rebuild loop needs updating to match) "
            f"or something upstream is wrong -- failing loudly rather than silently mishandling it.")


    snn_disc = nn.Sequential(*[_rebuild_layer(m) for m in snn_seq])

    w_idx = 0
    for m in snn_disc.modules():
        if isinstance(m, nn.Linear):
            # disc_weights[w_idx] is Conv2d-shaped, (out_channels, in_channels, 1, 1) --
            # DynapcnnNetwork converted our Linear to a 1x1 Conv2d internally (see
            # module comment above). Flatten back down to Linear's (out, in) shape.
            m.weight.data.copy_(torch.flatten(disc_weights[w_idx], start_dim=-3))
            w_idx += 1

    t_idx = 0
    for m in snn_disc.modules():
        if isinstance(m, sl.IAFSqueeze):
            # .data.copy_(), not direct assignment -- spike_threshold is a
            # registered nn.Parameter (confirmed directly: a plain assignment
            # here raises "cannot assign 'torch.FloatTensor' as parameter
            # spike_threshold" -- the old script's version of this exact line
            # had the same bug, caught here by actually running this against
            # a real IAFSqueeze layer rather than trusting the old code as-is).
            m.spike_threshold.data.copy_(disc_thresholds[t_idx])
            # BUG FIX: min_v_mem was never copied. _rebuild_layer() builds each layer with
            # min_v_mem=float(m.min_v_mem) -- the ORIGINAL, un-discretized layer's floor
            # (-1, i.e. -1 x a threshold of 1.0) -- while spike_threshold is overwritten
            # here with the chip's discretized value (320, 389, 379, 261). DynapcnnNetwork's
            # discretization scales BOTH by the same factor (the chip-bound layers print
            # min_v_mem = -320, -389, -379, -261, i.e. -1 x their own threshold), so snn_disc
            # was left with a membrane floor of -1 against thresholds ~300: about 0.3% of
            # threshold instead of 100%, i.e. essentially no room for the negative
            # (inhibitory) membrane excursions the trained network relies on. Visible in a real
            # run's own printout (snn_disc min_v_mem=-1 vs dynapcnn_net min_v_mem=-320), and in
            # its behaviour: discretized scored RMSE 51.70 / CC 0.563 -- WORSE than the chip it
            # is meant to approximate (43.65 / 0.711), and fired a uniform ~1.6-1.7x torch's rate
            # at every layer (an over-firing signature of a missing inhibition floor). The
            # discretized twin is what separates "quantization cost" from "hardware effects",
            # so it has to actually match the chip-bound network.
            m.min_v_mem.data.copy_(torch.as_tensor(disc_min_v_mem[t_idx], dtype=m.min_v_mem.dtype))
            t_idx += 1

    snn_disc.eval()
    print("\nsnn_disc (discretized, same topology as snn_seq):")
    print(snn_disc)

    # ---------------------------------------------------------------------------
    # 12. SpeckSim -- event-driven CPU simulator of the actual chip
    #     architecture, for testing without hardware access.
    #
    # RESOLVED (was flagged as an open question when this was first written):
    # passing dynapcnn_net itself (matching the old script) is wrong. Confirmed
    # directly from a real run's warnings: from_sequential(dynapcnn_net, ...)
    # produced "Layer with name: _dynapcnn_module ... is ignored", "...
    # _dynapcnn_module._dynapcnn_layers ... is ignored", "...
    # _dynapcnn_module.merge_layer ... is ignored" -- those three names are
    # DynapcnnNetwork's own private internal wrapper attributes (the
    # underscore prefix, and _dynapcnn_layers being the ModuleDict that
    # actually HOLDS every real Conv2d/IAFSqueeze layer), not the network
    # itself. from_sequential() doesn't know how to interpret a
    # DynapcnnNetwork instance -- it was tracing dynapcnn_net as a generic
    # nn.Module, finding only its bookkeeping wrappers, and silently
    # building a SpecSim network with none of the real layers mapped in at
    # all. That is consistent with (and a very plausible cause of) the
    # segfault this produced on first actual use: running events through a
    # simulated network the C++ backend never received real layers for is
    # exactly the kind of malformed state that crashes rather than raising a
    # clean Python exception.
    #
    # Fixed to match sinabs' own documented usage exactly: a plain sinabs
    # Sequential, not a DynapcnnNetwork wrapper. Uses snn_disc (the
    # DISCRETIZED, quantized model), not snn_seq -- SpecSim's whole purpose
    # is simulating what the ACTUAL CHIP would do, which runs on quantized
    # weights, so simulating the full-precision model would answer a
    # different question than the one this is actually for. Flatten()
    # prepended for the same reason as DynapcnnNetwork's own input shape
    # earlier (section 9's comment) -- from_sequential() traces with an
    # (N, C, H, W)-shaped dummy input too, and snn_disc has the same bare-
    # Linear-first structure that needed it there.
    # ---------------------------------------------------------------------------
    snn_disc_for_specksim = nn.Sequential(nn.Flatten(), *snn_disc)
    specksim_snn = from_sequential(snn_disc_for_specksim, input_shape=DYNAPCNN_INPUT_SHAPE)
    n_spike_layers = sum(1 for m in snn_seq if isinstance(m, sl.IAFSqueeze))
    print(f"\nSpecSim ready ({n_spike_layers} spiking layers)")
else:
    print("\nNEEDS_DYNAPCNN=False (only 'torch' selected via --models) -- skipping "
          "DynapcnnNetwork conversion, discretization, and SpecSim construction "
          "entirely (section 9-12). Add 'discretized' and/or 'specksim' and/or "
          "'speck' to --models to include them.")



# ---------------------------------------------------------------------------
# 13. Compare torch / discretized / SpeckSim / real Speck hardware
#
# Same sandbox caveat as section 9-12: samna/dynapcnn couldn't be
# installed or run here, so the actual hardware/simulator calls below are
# researched against sinabs' current documentation but NOT run end to
# end. Treat this as a draft to run and debug on your own system.
#
# THE MOST IMPORTANT CHANGE FROM THE OLD SCRIPT, and worth understanding
# before running any of this: the old script's decode_spikes() decodes
# each timestep's RAW spike output independently, with no memory across
# timesteps at all. This project's actual model maintains a PERSISTENT,
# DECAYING EMA across the whole trial (SNN_Speck.forward()'s own
# temporal_ema = decay * temporal_ema + x, decoded fresh each step) --
# confirmed directly, this is not a subtle difference: on a synthetic
# 10-timestep sequence, per-timestep-only decoding diverged from the
# correct EMA-accumulated decoding by up to 0.2 on a [0,1]-scaled
# prediction. Using the old, memoryless decode here would mean torch,
# discretized, SpeckSim, and real Speck predictions could all still
# AGREE WITH EACH OTHER while all being wrong relative to what this
# model was actually trained and evaluated to do -- a silent, severe
# correctness bug that a passing cross-implementation comparison would
# not surface. Every runner below reuses this project's own
# decode_from_ema(), maintaining the same persistent EMA state across
# timesteps that run_torch_model() (section 5) already does and has
# already been verified against model()'s own real predictions.
#
# events_from_frame() also has a real fix versus the old script: the old
# version (torch.where(frame > 0)) emits exactly one event per nonzero
# location, silently losing count information for any channel that fires
# more than once in a single 4ms bin -- a real possibility under this
# project's MultiSpike training convention (confirmed throughout
# model_bmi.py). Fixed to emit one event per actual spike count.
#
# The vmem_read mechanism from the old script (calculate_neuron_address,
# ReadNeuronValue events) is REMOVED entirely -- traced through the old
# script's own run_speck_model() and confirmed its result was read and
# immediately discarded, never used for anything. Dead code, not a
# needed step.
#
# Visualization: the old script's 7-panel 2D image-canvas animation
# (place_cross/create_images on a fixed frame) is replaced with
# reconstructed-position comparison, reusing this script's own
# make_position_frames()/plot_sample() -- consistent with section 7's
# same reasoning: this pipeline decodes velocity, not absolute position
# on a fixed canvas, so reconstructed position is the natural analogue,
# extended here to overlay all four implementations at once rather than
# just one.
# ---------------------------------------------------------------------------
from collections import defaultdict  # `import time` already done at module top

if RUN_ON_SPECK_HARDWARE:
    import samna
    import sinabs.backend.dynapcnn.io as sio
    from sinabs.backend.dynapcnn.chip_factory import ChipFactory

    DEVKIT_NAME = "speck2fdevkit:0"
    devkit = sio.open_device(DEVKIT_NAME)
    chip_factory = ChipFactory("speck2fdevkit")

    # --- Power monitoring setup -- VERIFIED against sinabs' own official
    # power-monitoring guide (https://sinabs.readthedocs.io/v2.0.3/speck/
    # notebooks/power_monitoring.html), not just researched from API docs
    # like the earlier revision of this code. That earlier revision had
    # TWO real bugs this fixes:
    #   1. Power events come from a DEDICATED buffer node
    #      (samna.BasicSinkNode_unifirm_modules_events_measurement),
    #      wired into samna_graph via power_monitor.get_source_node() --
    #      NOT from calling .get_events() directly on power_monitor
    #      itself, which doesn't have that method at all (confirmed
    #      directly: 'samna.boards.common.power.PowerMonitor' object has
    #      no attribute 'get_events').
    #   2. This branch MUST be added to samna_graph BEFORE
    #      samna_graph.start() is called -- matching the docs' own
    #      example, which builds every branch first, then starts the
    #      graph once at the end. See integrate_power_events()'s own
    #      docstring for the corresponding VALUE-units fix.
    stop_watch = devkit.get_stop_watch()
    power_monitor = devkit.get_power_monitor()
    power_source_node = power_monitor.get_source_node()
    power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()
    POWER_SAMPLE_RATE_HZ = 100  # samples/sec while measuring -- coarse enough that
    # integrate_power_events() uses the trapezoid rule, not a rectangle rule, to
    # integrate between readings (see that function's own docstring for why)

    samna_graph = samna.graph.EventFilterGraph()
    input_buffer_node = samna.BasicSourceNode_speck2f_event_input_event()
    sink_node = samna.BasicSinkNode_speck2f_event_output_event()
    samna_graph.sequential([input_buffer_node, devkit.get_model_sink_node()])
    samna_graph.sequential([devkit.get_model_source_node(), sink_node])
    samna_graph.sequential([power_source_node, power_buffer_node])  # power branch
    samna_graph.start()

    # Needed for power events to carry real timestamps -- without this,
    # every event's .timestamp reads 0 (this is the docs' own stated
    # symptom of skipping it: "timestamps are all zeros, can't plot
    # power vs. time, you might need to update the firmware"). Doing
    # this ONCE here, not per run_speck_model() call, since it's a
    # devkit-level setting, not a per-trial one.
    stop_watch.set_enable_value(True)

    devkit_cfg = dynapcnn_net.make_config(device=DEVKIT_NAME)
    print(f"Chip layer ordering: {dynapcnn_net.chip_layers_ordering}")

    # This project injects MUA spike events directly (input_buffer_node /
    # raster_to_events(), inside run_speck_model() below) -- it never reads
    # from the chip's physical DVS camera at all. pass_sensor_events=False
    # alone does NOT prevent the config validator from checking shape
    # compatibility for an ENABLED dvs_layer destination -- confirmed
    # directly against a real run: wiring destinations[0] to the first
    # network layer (matching the old script, which genuinely did use the
    # DVS camera for its own image-based task) made the validator compare
    # that layer's expected (96, 1, 1) input against the DVS sensor's
    # fixed, physical 128x128x2 output shape -- "Input space feature count
    # does not match target input feature count. 2 versus 96" -- and
    # apply_configuration() raised before the network was ever usable.
    # Not wiring a destination at all avoids this entirely; the DVS
    # pathway isn't needed here regardless of whether it's enabled.
    devkit_cfg.dvs_layer.pass_sensor_events = False

    # Reset-to-zero on-chip, matching this project's training-time reset_type=hard
    # convention (see model_bmi.py's NeuronFactory) -- on-chip behavior needs to
    # match what the model was actually trained under.
    for i in range(len(dynapcnn_net.chip_layers_ordering)):
        devkit_cfg.cnn_layers[dynapcnn_net.chip_layers_ordering[i]].return_to_zero = True
        devkit_cfg.cnn_layers[dynapcnn_net.chip_layers_ordering[i]].monitor_enable = True

    devkit.get_model().apply_configuration(devkit_cfg)
else:
    print("\n'speck' not in --models -- skipping device connection, hardware config, "
          "and everything else that needs the physical devkit.")


def integrate_power_events(power_events):
    """Integrates samna power-measurement events into total energy, in
    Joules, for whatever span of events was collected (one
    run_speck_model() call here -- see that function's own use of this).

    VERIFIED against sinabs' own official power-monitoring guide
    (https://sinabs.readthedocs.io/v2.0.3/speck/notebooks/
    power_monitoring.html) -- NOT researched-but-unverified anymore, per
    the earlier revision's own caveat. That earlier revision had the
    UNITS WRONG: each event has `.channel` (an int power-rail index --
    0=io, 1=ram, 2=logic, 3=pixel digital/VDDD, 4=pixel analog/VDDA, per
    the docs) and `.value` (instantaneous power draw on that channel, in
    WATTS -- confirmed directly from the docs' own worked example:
    `avg_power = ... * 1e6` labeled as microwatts, meaning the raw value
    is already in watts, NOT milliwatts as this function previously
    assumed) and `.timestamp` (in MICROSECONDS -- this assumption WAS
    already correct, confirmed via the docs' own `ax.set_xlabel("time
    (us)")`). If a future samna version changes either convention,
    `print(power_events[:5])` right after one real measurement is the
    fastest way to re-check, and the one remaining divisor below (1e6
    for us->s) is what to fix.

    Sums power across EVERY channel present (io + logic + ram + ... --
    total chip draw, not just one rail), then integrates each channel's
    own timestamp-ordered sequence via the TRAPEZOID rule (average of
    two consecutive readings x the time between them), not a rectangle
    rule -- PowerMonitor's sample rate (POWER_SAMPLE_RATE_HZ) is coarse
    relative to a single 4ms timestep, and treating each reading as
    constant until the next one arrives would bias the estimate
    depending on whether power happened to be rising or falling between
    samples, rather than averaging the two.
    """
    by_channel = defaultdict(list)
    for ev in power_events:
        by_channel[ev.channel].append((ev.timestamp, ev.value))

    total_energy_j = 0.0
    for channel, readings in by_channel.items():
        readings.sort(key=lambda r: r[0])
        for (t0, p0), (t1, p1) in zip(readings[:-1], readings[1:]):
            dt_s = (t1 - t0) / 1e6   # microseconds -> seconds (confirmed)
            avg_power_w = (p0 + p1) / 2   # already in WATTS (confirmed) -- no /1000 needed
            total_energy_j += avg_power_w * dt_s
    return total_energy_j


def summarize_power_events(power_events, loop_duration_s: float, sample_rate_hz: float) -> dict:
    """Energy for one run_speck_model() call, WITHOUT trusting the events' timestamps for the
    time base, plus everything needed to audit that choice afterwards.

    WHY: integrate_power_events() multiplies each reading by the timestamp gap to its
    neighbour. In every real run so far the power monitor delivered exactly 500.0 events per
    second of decode loop -- 100 Hz (POWER_SAMPLE_RATE_HZ) x 5 channels, to within 0.02% over
    five runs, i.e. it sampled for the WHOLE loop -- yet those events' timestamps spanned only
    0.006-0.045 s of a ~33 s loop (0.02-0.12%). The power VALUES are fine (they are
    instantaneous watts); the TIME BASE is not, so the timestamp integral came out ~1000x too
    small (15-113 uJ for a loop that, at the ~1-3 mW the same readings imply, costs tens of mJ).
    Energy = mean power x duration, and the duration is independently known: latency_s is
    measured with perf_counter over exactly this loop (and the sample count / rate agrees with
    it). So: energy_j_corrected = (sum over channels of that channel's mean power) x
    loop_duration_s. Assumes power is roughly stationary over the loop.

    This is TOTAL chip draw over the loop -- idle/static power included, nothing subtracted,
    and it is the energy of THIS loop (which runs faster than the 4 ms native step: real-time
    streaming of the same trial would take longer), not an idle-subtracted energy per
    inference. total_mean_power_w is returned so any of those conventions can be derived from
    the saved JSON afterwards instead of re-running the chip.

    The raw timestamp behaviour is recorded (median gap between consecutive samples on a
    channel, and the fraction of exactly-zero gaps -- all-zero would mean events are stamped at
    flush time) so the cause can be pinned down from any log. Expected for a healthy monitor:
    a median gap of 1e6 / sample_rate_hz microseconds (10000 at 100 Hz).
    """
    out = {"n_events": len(power_events), "sample_rate_hz_assumed": float(sample_rate_hz),
           "loop_s": float(loop_duration_s)}
    if not power_events:
        out.update({"n_channels": 0, "n_samples_per_channel": 0, "per_channel_mean_power_w": {},
                    "total_mean_power_w": 0.0, "implied_duration_s": 0.0, "timestamp_span_s": 0.0,
                    "median_raw_timestamp_step": None, "frac_zero_timestamp_step": None,
                    "sample_count_matches_loop": None, "timestamps_match_loop": None,
                    "energy_j_timestamp_integral": 0.0, "energy_j_corrected": 0.0})
        return out
    by_channel = defaultdict(list)
    for ev in power_events:
        by_channel[ev.channel].append((ev.timestamp, ev.value))
    per_channel_mean_w, n_per_channel, steps = {}, {}, []
    all_ts = []
    for ch, readings in by_channel.items():
        readings.sort(key=lambda r: r[0])
        ts = [float(t) for t, _ in readings]
        per_channel_mean_w[str(ch)] = float(np.mean([float(p) for _, p in readings]))
        n_per_channel[str(ch)] = len(readings)
        all_ts.extend(ts)
        if len(ts) > 1:
            steps.extend(np.diff(ts).tolist())
    total_mean_power_w = float(sum(per_channel_mean_w.values()))
    n_samples = int(np.median(list(n_per_channel.values())))
    implied_duration_s = n_samples / float(sample_rate_hz)
    span_s = (max(all_ts) - min(all_ts)) / 1e6
    tol = 0.10 * float(loop_duration_s)
    out.update({
        "n_channels": len(by_channel), "n_samples_per_channel": n_samples,
        "per_channel_mean_power_w": per_channel_mean_w, "total_mean_power_w": total_mean_power_w,
        "implied_duration_s": implied_duration_s, "timestamp_span_s": float(span_s),
        "median_raw_timestamp_step": float(np.median(steps)) if steps else None,
        "frac_zero_timestamp_step": float(np.mean(np.asarray(steps) == 0)) if steps else None,
        "sample_count_matches_loop": bool(abs(implied_duration_s - loop_duration_s) <= tol),
        "timestamps_match_loop": bool(abs(span_s - loop_duration_s) <= tol),
        "energy_j_timestamp_integral": float(integrate_power_events(power_events)),
        "energy_j_corrected": total_mean_power_w * float(loop_duration_s)})
    return out


def events_from_frame(frame: torch.Tensor) -> np.ndarray:
    """Binary/count [1, C, H, W] -> structured numpy event array for
    SpeckSim/hardware. Emits ONE EVENT PER ACTUAL SPIKE COUNT (not one
    event per nonzero location) -- see module comment above for why this
    matters under this project's MultiSpike training convention.

    Vectorized (np.repeat instead of a Python-level loop + list.extend) --
    confirmed identical output against the original across 200 random
    sparse frames plus the all-zero case, and confirmed this only saves
    ~9% per call (already ~17us/call given how sparse this project's
    input actually is -- roughly 4ms total across an entire 10-trial
    comparison run). Kept anyway since it's a strict improvement with no
    downside, but this is NOT where SpecSim's real runtime cost is --
    see the new per-implementation latency metrics in the evaluation
    loop below for actually locating that.
    """
    dtype = [("t", np.uint32), ("p", np.uint32), ("y", np.uint32), ("x", np.uint32)]
    frame_np = frame.squeeze(0).numpy()
    nz_c, nz_y, nz_x = np.nonzero(frame_np)
    if len(nz_c) == 0:
        return np.array([], dtype=dtype)
    counts = np.round(frame_np[nz_c, nz_y, nz_x]).astype(int)
    rep_c = np.repeat(nz_c, counts)
    rep_y = np.repeat(nz_y, counts)
    rep_x = np.repeat(nz_x, counts)
    events = np.zeros(len(rep_c), dtype=dtype)
    events["t"] = 0
    events["p"] = rep_c
    events["y"] = rep_y
    events["x"] = rep_x
    return events


def set_all_v_mem_to_zeros(cfg_builder, samna_device, layer_id: int) -> None:
    mod = cfg_builder.get_samna_module()
    layer_constraint = cfg_builder.get_constraints()[layer_id]
    events = []
    for i in range(layer_constraint.neuron_memory):
        event = mod.event.WriteNeuronValue()
        event.address = i
        event.layer = layer_id
        event.neuron_state = 0
        events.append(event)

    temporary_source_node = cfg_builder.get_input_buffer()
    temporary_graph = samna.graph.EventFilterGraph()
    temporary_graph.sequential([temporary_source_node, samna_device.get_model().get_sink_node()])
    temporary_graph.start()
    temporary_source_node.write(events)
    time.sleep(1)
    temporary_graph.stop()


def run_specksim_model(model, input_tensor, n_layers):
    """input_tensor: (T, C, H, W). Returns (predictions (T, 2) SCALED
    space, per-layer spike counts, latency_s, energy_j, energy_method)
    -- SAME persistent-EMA decode as run_torch_model() (section 5), see
    module comment above for why. latency_s/energy_j: same scoping
    principle as run_torch_model() -- measured over ONLY the decode loop
    below, not reset_states()/clear_monitors()/add_monitor() before it
    or reset_states() after."""
    model.reset_states()
    model.clear_monitors()
    for i in range(n_layers):
        model.add_monitor(i)

    T = input_tensor.shape[0]
    spike_counts = {i: [] for i in range(n_layers)}
    predictions = []
    ema_stages = [torch.zeros(1, N_BINS * 2) for _ in range(TEMPORAL_DECAY_STAGES)]

    with EnergyMeter() as em:
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in tqdm(range(T), desc="SpeckSim"):
                frame = input_tensor[t].unsqueeze(0)
                out_events = model(events_from_frame(frame))

                spikes = torch.zeros(1, N_BINS * 2)
                for spk in out_events:
                    feat = spk[-1]
                    if feat < N_BINS * 2:
                        spikes[0, feat] += 1

                ema = ema_cascade_update(ema_stages, spikes, TEMPORAL_DECAY)
                predictions.append(decode_from_ema(ema).squeeze(0))
                for i in range(n_layers):
                    spike_counts[i].append(len(model.monitors[i]["sink"].get_events()))
                progress_checkpoint("specksim", t, T, t0)

    model.reset_states()
    return torch.stack(predictions), spike_counts, em.latency_s, em.energy_j, em.energy_method


# Filled in by run_speck_model(): how many input events were actually sent to the chip versus
# how many input spike COUNTS the raster held. The discretized twin consumes the counts directly;
# the chip only sees whatever chip_factory.raster_to_events() turns them into, and nothing else
# in this script verified that a count of 2 or 3 in one bin survives as 2 or 3 events rather than
# being merged into one. Kept out of the function's return value on purpose (callers unpack it).
SPECK_INPUT_AUDIT = {}
POWER_AUDIT = {}   # filled by run_speck_model(): see summarize_power_events()


def run_speck_model(input_tensor, time_steps):
    """input_tensor: (T, C, H, W). Returns (predictions (T, 2) SCALED
    space, {core_idx: [spike counts]}, latency_s, energy_j,
    energy_method) -- SAME persistent-EMA decode as every other runner
    here. energy_j is REAL on-chip energy for this call, from the
    devkit's own PowerMonitor telemetry (see integrate_power_events())
    -- not a host-side estimate; this is the actual number the whole
    point of deploying to hardware is for. energy_method is always
    'chip_power_monitor', returned rather than hardcoded at the call
    site so every runner's return signature matches
    run_torch_model()'s/run_specksim_model()'s exactly.

    latency_s AND energy_j span the SAME window -- the timestep loop
    below (timed from right before it starts to right after it ends),
    NOT the v_mem-reset settling time above it. Requested directly:
    resetting isn't decoding, so it shouldn't count toward "how long
    does decoding take" -- the v_mem writes across every core plus the
    1s settle can be a real, multi-second cost on top of the actual
    per-timestep decode work, which would otherwise silently inflate
    this implementation's reported latency (and, if it were included,
    energy) relative to what deployment-time streaming decode actually
    costs once the chip is already running.
    """
    wait_time = args.speck_wait_time
    cores = dynapcnn_net.chip_layers_ordering

    predictions = []
    spike_counts = {c: [] for c in cores}
    ema_stages = [torch.zeros(1, N_BINS * 2) for _ in range(TEMPORAL_DECAY_STAGES)]
    input_events_sent = 0          # None once it turns out raster_to_events() isn't sized
    input_counts_expected = 0.0

    sink_node.clear_events()
    for layer in cores:
        set_all_v_mem_to_zeros(chip_factory.get_config_builder(), devkit, layer)
    time.sleep(1)

    # Clear any power events that accumulated in the buffer between the
    # last call and this one (e.g. idle-power readings while the v_mem
    # reset above was settling) -- start_auto_power_measurement() does
    # NOT clear the buffer itself, it only starts producing NEW events
    # into it (confirmed via the docs' own idle-power example, which
    # calls power_buffer_node.get_events() once immediately before
    # start_auto_power_measurement() specifically to discard stale data).
    power_buffer_node.get_events()

    # Optional IDLE BASELINE: same configured network, v_mem already zeroed above, NO input events written.
    # Measured with the identical start/stop/get_events sequence as the real run below, then the buffers are
    # cleared again so none of it leaks into the real measurement.
    idle_summary = None
    if args.idle_baseline_seconds > 0:
        power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
        time.sleep(args.idle_baseline_seconds)
        power_monitor.stop_auto_power_measurement()
        idle_summary = summarize_power_events(power_buffer_node.get_events(), args.idle_baseline_seconds,
                                              POWER_SAMPLE_RATE_HZ)
        sink_node.clear_events()
        power_buffer_node.get_events()

    power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
    t0 = time.perf_counter()
    for t in tqdm(range(time_steps), desc="Speck hardware"):
        timeframe = input_tensor[t].unsqueeze(0)
        # raster_to_events() (inside sinabs' own chip_factory.py) crashes with
        # "stack expects a non-empty TensorList" when a frame has ZERO spikes
        # across every channel -- confirmed directly: torch.stack(sorted([]))
        # raises exactly that message. This is an ordinary, expected situation
        # with sparse MUA input (some individual 4ms timesteps across a
        # 260-step trial having no spikes at all across all 96 channels is
        # normal, not a data problem), not something raster_to_events()
        # itself tolerates. Nothing to send this timestep if the frame is
        # all-zero -- skip straight to waiting/reading output, since the
        # chip's membrane potentials can still be decaying or producing
        # delayed spikes from earlier input regardless of new input arriving.
        if timeframe.any():
            event_stream = chip_factory.raster_to_events(timeframe, cores[0], dt=args.speck_raster_dt)
            input_counts_expected += float(timeframe.sum())
            if input_events_sent is not None:
                try:
                    input_events_sent += len(event_stream)
                except TypeError:
                    input_events_sent = None
            stop_watch.start(reset=True)
            input_buffer_node.write(event_stream)
        time.sleep(wait_time)

        raw_events = sink_node.get_events()
        spike_events = [ev for ev in raw_events if isinstance(ev, samna.speck2f.event.Spike)]

        per_core = defaultdict(int)
        for ev in spike_events:
            per_core[ev.layer] += 1
        for c in cores:
            spike_counts[c].append(per_core[c])

        output_spikes = [spk for spk in spike_events if spk.layer == cores[-1]]
        spikes = torch.zeros(1, N_BINS * 2)
        for feature in range(N_BINS * 2):
            spikes[0, feature] = sum(1 for spk in output_spikes if spk.feature == feature)

        ema = ema_cascade_update(ema_stages, spikes, TEMPORAL_DECAY)
        predictions.append(decode_from_ema(ema).squeeze(0))
        progress_checkpoint("speck", t, time_steps, t0)
    latency_s = time.perf_counter() - t0  # SAME window as energy below (t0 set right
    # before this loop, latency stopped right after it -- computed here, before
    # stop_auto_power_measurement()'s own call overhead, so it reflects purely the
    # decode loop's own duration)
    SPECK_INPUT_AUDIT.update({"input_counts_expected": input_counts_expected,
                              "input_events_sent": input_events_sent})
    if input_events_sent is None:
        print("  [speck input audit] unavailable: raster_to_events() did not return a sized sequence")
    else:
        _ratio = input_events_sent / input_counts_expected if input_counts_expected else float("nan")
        SPECK_INPUT_AUDIT["ratio_events_per_count"] = _ratio
        print(f"  [speck input audit] {input_events_sent} input events sent for "
              f"{input_counts_expected:.0f} input spike counts in the raster (ratio {_ratio:.3f}; "
              f"1.000 = every count reaches the chip as its own event, below 1 = counts above 1 "
              f"are being merged or clipped on the way in)")
    power_monitor.stop_auto_power_measurement()
    power_events = power_buffer_node.get_events()  # NOT power_monitor.get_events() -- see
    # integrate_power_events()'s docstring for why that doesn't exist

    # Diagnostic: does the power-sampling window actually span this whole
    # call, or did something (buffer clearing, timing, an empty channel)
    # cause it to only capture a fraction of it? At POWER_SAMPLE_RATE_HZ,
    # a healthy call should have roughly
    # (call_duration_s * POWER_SAMPLE_RATE_HZ * n_channels) events -- if
    # the printed window is much shorter than this call's own wall-clock
    # time (timed separately, in the eval loop below), that's a strong
    # sign the energy_j this call returns is UNDER-measuring, not just
    # imprecise -- worth checking before trusting a suspiciously small
    # energy_j number the way an implausibly LARGE one would also be
    # worth checking (see energy_meter.py's own thread-pinning docstring
    # for that side of this).
    if power_events:
        span_s = (max(ev.timestamp for ev in power_events) -
                  min(ev.timestamp for ev in power_events)) / 1e6
        print(f"  [power diagnostic] {len(power_events)} events across "
              f"{len(set(ev.channel for ev in power_events))} channel(s), "
              f"spanning {span_s:.3f}s of sampled timestamps (vs. {latency_s:.3f}s "
              f"decode-loop latency)")
    else:
        print("  [power diagnostic] WARNING: zero power events collected this call -- "
              "energy_j will be 0.0, not a measurement of anything")
    _ps = summarize_power_events(power_events, latency_s, POWER_SAMPLE_RATE_HZ)
    POWER_AUDIT.clear()
    POWER_AUDIT.update(_ps)
    energy_j = _ps["energy_j_corrected"]
    if idle_summary is not None:
        if idle_summary["n_events"] == 0:
            print("  [power] WARNING: the idle baseline collected zero power events -- dynamic power not computed")
        else:
            _idle_w = idle_summary["total_mean_power_w"]
            _dyn_w = _ps["total_mean_power_w"] - _idle_w
            POWER_AUDIT.update({
                "idle_baseline_seconds": float(args.idle_baseline_seconds),
                "idle_mean_power_w": _idle_w,
                "idle_per_channel_mean_power_w": idle_summary["per_channel_mean_power_w"],
                "idle_sample_count_matches": idle_summary["sample_count_matches_loop"],
                # active minus idle, per rail and in total. Can be slightly NEGATIVE when the workload is below
                # the measurement noise / drift between the two windows -- kept raw, not clipped, so that
                # is visible rather than hidden.
                "dynamic_mean_power_w": _dyn_w,
                "dynamic_per_channel_mean_power_w": {ch: _ps["per_channel_mean_power_w"].get(ch, 0.0) - w
                                                     for ch, w in idle_summary["per_channel_mean_power_w"].items()}})
            print(f"  [power] idle baseline {_idle_w * 1e3:.3f} mW over {args.idle_baseline_seconds:g}s (no input) | active "
                  f"{_ps['total_mean_power_w'] * 1e3:.3f} mW | DYNAMIC (active - idle) {_dyn_w * 1e3:+.3f} mW = "
                  f"{_dyn_w / _ps['total_mean_power_w'] * 100:+.1f}% of the active draw, {_dyn_w * 0.004 * 1e6:+.3f} uJ per 4 ms step")
    if power_events:
        _step = _ps["median_raw_timestamp_step"]
        print(f"  [power] mean chip draw {_ps['total_mean_power_w'] * 1e3:.3f} mW "
              f"({_ps['n_samples_per_channel']} samples/channel x {_ps['n_channels']} channels = "
              f"{_ps['implied_duration_s']:.2f}s at {POWER_SAMPLE_RATE_HZ} Hz vs the {latency_s:.2f}s measured loop: "
              f"{'consistent' if _ps['sample_count_matches_loop'] else 'INCONSISTENT'})")
        print(f"  [power] energy = mean draw x loop time = {energy_j * 1e3:.2f} mJ for this loop "
              f"(the old timestamp integral gave {_ps['energy_j_timestamp_integral'] * 1e6:.1f} uJ; its timestamps span "
              f"{_ps['timestamp_span_s']:.3f}s = "
              f"{'consistent with the loop' if _ps['timestamps_match_loop'] else 'NOT consistent with the loop, so not used'})")
        print(f"  [power] raw timestamp step between consecutive samples: median {_step} "
              f"(a healthy {POWER_SAMPLE_RATE_HZ} Hz monitor in microseconds gives {1e6 / POWER_SAMPLE_RATE_HZ:.0f}), "
              f"{100 * _ps['frac_zero_timestamp_step']:.1f}% of steps are exactly zero")

    return torch.stack(predictions), spike_counts, latency_s, energy_j, "chip_power_monitor"


def make_comparison_frames(preds_by_impl: dict, target_scaled: np.ndarray, img_size: int = PLOT_IMG_SIZE):
    """preds_by_impl: {impl_name: (T,2) SCALED predictions}. Returns
    {impl_name: (T,2) pixel-space reconstructed positions} plus the
    ground truth's own pixel-space positions, all sharing ONE common
    scale (built from every trace together) so relative sizes stay
    meaningful across implementations, matching make_position_frames()'s
    own reasoning (section 7) extended to more than two traces."""
    vel_true_phys = unscale_velocity(target_scaled, V_LO, V_HI, V_MARGIN)
    pos_true = reconstruct_path(np.zeros(2), vel_true_phys, STEP_TIME)

    pos_by_impl = {}
    for impl, pred_scaled in preds_by_impl.items():
        vel_phys = unscale_velocity(pred_scaled.numpy(), V_LO, V_HI, V_MARGIN)
        pos_by_impl[impl] = reconstruct_path(np.zeros(2), vel_phys, STEP_TIME)

    all_pos = np.concatenate([pos_true] + list(pos_by_impl.values()), axis=0)
    lo, hi = all_pos.min(axis=0), all_pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def to_pixels(pos):
        return ((pos - lo) / span * (img_size - 1)).astype(int)

    return {impl: to_pixels(pos) for impl, pos in pos_by_impl.items()}, to_pixels(pos_true)


def plot_comparison(trial_idx, pos_by_impl_px, pos_true_px, img_size=PLOT_IMG_SIZE, save_dir=PLOT_SAVE_DIR):
    colors = IMPL_COLORS

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_title(f"Sample {trial_idx}: true (black) vs. every implementation")

    true_line, = ax.plot([], [], color='black', linewidth=1.5, label='true')
    lines = {"true": true_line}
    for impl in pos_by_impl_px:
        line, = ax.plot([], [], color=colors.get(impl, 'gray'), linewidth=1.2,
                         linestyle='--', label=impl)
        lines[impl] = line
    ax.legend(loc='upper right', fontsize=8)

    def update(t):
        true_line.set_data(pos_true_px[:t + 1, 0], pos_true_px[:t + 1, 1])
        for impl, pos_px in pos_by_impl_px.items():
            lines[impl].set_data(pos_px[:t + 1, 0], pos_px[:t + 1, 1])
        return list(lines.values())

    ani = FuncAnimation(fig, update, frames=pos_true_px.shape[0], interval=100)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ani.save(f"{save_dir}/comparison_{trial_idx}.gif", writer="pillow", fps=10)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 13b. Crosshair-style comparison -- a second, additional visualization of
# the SAME reconstructed positions from make_comparison_frames(), styled
# after this project's earlier reference notebook code (place_cross(),
# create_images(), the multi-panel + combined-RGB-overlay figure): a
# moving crosshair marker per implementation on its own black-background
# panel, plus one combined, color-coded overlay panel where every trace's
# crosshair is additively blended onto separate RGB channel(s) -- rather
# than plot_comparison()'s accumulating line traces on a shared white
# background. This does NOT replace plot_comparison() -- both are kept,
# since they show different things (full path shape vs. instantaneous
# per-implementation agreement).
#
# One thing NOT ported from the reference: its "Input Spikes" panel
# (two-channel event-camera frames rendered as blue/red overlays). This
# project's own input is 96-channel MUA rate data, which has no
# meaningful (H, W) image representation the way a 2-channel DVS frame
# does -- forcing it into that shape would just be noise, not signal, so
# it's left out rather than faked.
#
# Fixes a real bug in the reference's static (non-animated) frame-init
# code -- it assigned per-implementation channels with `=`, so each
# later implementation silently overwrote the previous one's channel
# contribution instead of blending with it (its animated update()
# function did this correctly with `+=`; only the very first frame was
# affected). Verified directly here: two traces at the exact same pixel
# both remain visible (additively blended), not one clobbering the other.
# ---------------------------------------------------------------------------
def place_cross(image: np.ndarray, x: int, y: int, img_size: int) -> np.ndarray:
    """Draws a small 5-pixel plus/cross marker at (x, y) on `image` (a
    black-background (img_size, img_size) array) -- same shape as this
    project's earlier reference code's place_cross(), generalized to
    img_size instead of a hardcoded 32. Clips at the frame boundary
    rather than raising if x/y land at the very edge (verified directly:
    a cross at a corner correctly keeps only its in-bounds arms)."""
    for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        px, py = x + dx, y + dy
        if 0 <= px < img_size and 0 <= py < img_size:
            image[py, px] = 1.0
    return image


def make_crosshair_frame(pos_px_at_t: np.ndarray, img_size: int) -> np.ndarray:
    """pos_px_at_t: (2,) integer pixel position for ONE timestep. Returns
    one fresh (img_size, img_size) black-background frame with a
    crosshair at that position -- built fresh per call, not
    pre-generated for the whole trial up front, to keep memory bounded
    regardless of trial length (matching plot_comparison()'s own
    incremental-update convention, rather than the reference code's
    generate-every-frame-up-front approach, which doesn't scale well to
    this project's longer continuous/huge-dataset trials)."""
    frame = np.zeros((img_size, img_size), dtype=np.float64)
    return place_cross(frame, int(pos_px_at_t[0]), int(pos_px_at_t[1]), img_size)


# RGB channel(s) per trace for the combined overlay panel. GT + up to 4
# implementations sharing 3 channels -- specksim/speck double up on a
# channel with true/torch specifically, matching the reference's
# intended red/green/blue/yellow/magenta scheme (fixed to blend
# correctly rather than overwrite, per the note above).
CROSSHAIR_OVERLAY_CHANNELS = {
    "true":        [0],        # red
    "torch":       [1],        # green
    "discretized": [2],        # blue
    "specksim":    [0, 1],     # yellow
    "speck":       [0, 2],     # magenta
}


def make_overlay_frame(pos_true_px_at_t: np.ndarray, pos_by_impl_px_at_t: dict,
                        img_size: int) -> np.ndarray:
    """Additively blends a crosshair per trace onto its assigned RGB
    channel(s) (CROSSHAIR_OVERLAY_CHANNELS), clipped to [0, 1] --
    overlapping crosshairs blend into a combined color (e.g. GT and
    torch agreeing exactly -> yellow) rather than one silently
    overwriting another's channel."""
    frame = np.zeros((img_size, img_size, 3), dtype=np.float64)
    single_true = make_crosshair_frame(pos_true_px_at_t, img_size)
    for ch in CROSSHAIR_OVERLAY_CHANNELS["true"]:
        frame[:, :, ch] += single_true
    for impl, pos_px in pos_by_impl_px_at_t.items():
        single = make_crosshair_frame(pos_px, img_size)
        for ch in CROSSHAIR_OVERLAY_CHANNELS.get(impl, [1]):
            frame[:, :, ch] += single
    return np.clip(frame, 0, 1)


def plot_crosshair_comparison(trial_idx, pos_by_impl_px, pos_true_px, img_size=PLOT_IMG_SIZE,
                               save_dir=PLOT_SAVE_DIR, label="position"):
    """Companion to plot_comparison() -- takes the SAME reconstructed,
    pixel-space positions (from make_comparison_frames()), rendered in
    the crosshair-panel style described above instead of accumulating
    line traces.

    `label` controls the title text and output filename ONLY -- the
    crosshair-drawing logic itself is completely agnostic to what the
    (x, y) pixel coordinates actually represent. Default "position"
    preserves this function's original behavior and filename exactly.
    Also reused directly for RAW VELOCITY crosshairs (label="velocity"),
    fed velocity_to_pixel()'s output instead of make_comparison_frames()'s
    reconstructed-position output -- no separate velocity-crosshair
    drawing code needed, since the mechanism doesn't care what space the
    points came from.
    """
    impls = list(pos_by_impl_px.keys())
    n_panels = 1 + len(impls) + 1  # groundtruth, each impl, combined overlay
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    panel_titles = ["Groundtruth"] + [impl.capitalize() for impl in impls] + ["Combined overlay"]
    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    gt_display = axes[0].imshow(np.zeros((img_size, img_size)), cmap='gray', vmin=0, vmax=1)
    impl_displays = {}
    for i, impl in enumerate(impls):
        impl_displays[impl] = axes[1 + i].imshow(
            np.zeros((img_size, img_size)), cmap='gray', vmin=0, vmax=1)
    overlay_display = axes[-1].imshow(np.zeros((img_size, img_size, 3)), vmin=0, vmax=1)

    def update(t):
        gt_display.set_data(make_crosshair_frame(pos_true_px[t], img_size))
        for impl in impls:
            impl_displays[impl].set_data(make_crosshair_frame(pos_by_impl_px[impl][t], img_size))
        pos_by_impl_at_t = {impl: pos_by_impl_px[impl][t] for impl in impls}
        overlay_display.set_data(make_overlay_frame(pos_true_px[t], pos_by_impl_at_t, img_size))
        return [gt_display] + list(impl_displays.values()) + [overlay_display]

    fig.suptitle(f"Sample {trial_idx}: crosshair {label}, per-implementation and combined")
    fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=pos_true_px.shape[0], interval=100)
    filename_prefix = "crosshair_comparison" if label == "position" else f"{label}_crosshair_comparison"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ani.save(f"{save_dir}/{filename_prefix}_{trial_idx}.gif", writer="pillow", fps=10)
    else:
        plt.show()
    plt.close(fig)


def velocity_to_pixel(v_phys, v_lo=V_LO, v_hi=V_HI, img_size=PLOT_IMG_SIZE):
    """v_phys: (T, 2) physical velocity. Maps [v_lo, v_hi] -> [0, img_size-1]
    pixel coordinates directly, using this project's own KNOWN, FIXED
    velocity bounds -- unlike reconstructed position (which has no fixed
    range and needs a trajectory-dependent normalization, see
    make_comparison_frames()), velocity's bounds are already established
    throughout this project, so nothing needs to be computed from this
    specific trial's own min/max. Clipped to [0, img_size-1] rather than
    letting a genuinely out-of-range prediction (a real possibility, not
    just clamped-in-training ground truth) wrap or error.
    """
    frac = (v_phys - v_lo) / (v_hi - v_lo)
    frac = np.clip(frac, 0.0, 1.0)
    return np.round(frac * (img_size - 1)).astype(int)


def plot_velocity_over_time(trial_idx, phys_by_impl, target_phys, save_dir=PLOT_SAVE_DIR):
    """RAW velocity over time -- vx and vy each in their own subplot,
    ground truth plus every implementation's own prediction, NO
    reconstruction/integration into position and no aggregating across
    implementations -- exactly what plot_comparison()/
    plot_crosshair_comparison() do NOT show (both operate on
    reconstruct_path()'s integrated position instead). Same color
    convention as plot_comparison() for visual consistency across every
    plot in this script.
    """
    colors = IMPL_COLORS
    T = target_phys.shape[0]
    time_s = np.arange(T) * STEP_TIME

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axis_labels = ["vx", "vy"]
    for row in range(2):
        ax = axes[row]
        ax.plot(time_s, target_phys[:, row], color='black', linewidth=1.0, label='true')
        for impl, phys in phys_by_impl.items():
            ax.plot(time_s, phys[:, row], color=colors.get(impl, 'gray'), linewidth=1.0,
                    alpha=0.8, linestyle='--', label=impl)
        ax.set_ylabel(axis_labels[row])
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f"Sample {trial_idx}: raw velocity over time, per-implementation")
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/velocity_over_time_{trial_idx}.png", dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_energy_summary(energy_samples, energy_methods, active_impls, save_dir=PLOT_SAVE_DIR):
    """Bar chart of mean per-trial energy (J) across every active
    implementation that has valid data. LOG-SCALE y-axis, deliberately --
    torch/discretized's MAC/ACC-based estimate, specksim's host energy
    (via EnergyMeter), and speck's real on-chip energy (via PowerMonitor)
    routinely differ by several orders of magnitude (in one real run:
    ~1608 J vs. ~0.037 J -- a >40,000x spread), which is an EXPECTED
    consequence of comparing an estimate/general-purpose CPU software
    against dedicated neuromorphic hardware, not a plotting bug. A linear
    axis would make the smaller bar(s) visually vanish -- exactly the
    same problem this project's other energy figure
    (plot_decoder_efficiency_aggregate.py's make_energy_bar_figure())
    already had to solve for KF/WF vs. LSTM/QRNN/SNN's own
    multi-order-of-magnitude spread.

    Same IMPL_COLORS mapping as every other plot in this script. Saved
    as a static PNG (fig.savefig), not a FuncAnimation -- no OOM risk
    regardless of trial length, unlike the position/velocity animations
    elsewhere in this file.
    """
    EPS = 1e-9  # log scale can't plot zero/negative -- floor rather than silently
    # dropping a real (if extremely small) measurement like speck's own

    names, means, err_lo, err_hi, labels = [], [], [], [], []
    for impl in active_impls:
        raw_samples = [e for e in energy_samples.get(impl, []) if e is not None]
        if len(raw_samples) == 0:
            print(f"  [plot_energy_summary] skipping {impl}: no valid energy readings")
            continue
        method = energy_methods.get(impl) or "n/a"

        if isinstance(raw_samples[0], dict):
            # torch/discretized: MAC/ACC estimator range (see
            # run_torch_model()'s own docstring). Bar height = mean of
            # (low, high) across trials; error bar = the low/high range
            # ITSELF (the estimator's SRAM-vs-DRAM uncertainty), not
            # trial-to-trial std -- a materially different, and more
            # honest, thing to show for a range-valued estimate.
            lows = np.array([e["low"] for e in raw_samples], dtype=float)
            highs = np.array([e["high"] for e in raw_samples], dtype=float)
            mean_low, mean_high = float(lows.mean()), float(highs.mean())
            mean = max((mean_low + mean_high) / 2, EPS)
            names.append(impl)
            means.append(mean)
            err_lo.append(min(mean - mean_low, mean - EPS))
            err_hi.append(mean_high - mean)
            labels.append(f"[{mean_low:.4g}, {mean_high:.4g}] J\nn={len(raw_samples)} ({method})")
            continue

        arr = np.array(raw_samples, dtype=float)
        mean = max(float(arr.mean()), EPS)
        std = float(arr.std()) if len(arr) > 1 else 0.0
        names.append(impl)
        means.append(mean)
        err_lo.append(min(std, mean - EPS))  # can't dip to/below zero on a log axis
        err_hi.append(std)
        labels.append(f"{arr.mean():.4g} J\nn={len(arr)} ({method})")

    if not names:
        print("  [plot_energy_summary] no implementation has valid energy data -- skipping figure")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(names))
    bar_colors = [IMPL_COLORS.get(n, 'gray') for n in names]
    ax.bar(x_pos, means, color=bar_colors, alpha=0.85, edgecolor='black')
    ax.errorbar(x_pos, means, yerr=[err_lo, err_hi], fmt='none',
                ecolor='black', elinewidth=1.2, capsize=4, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_ylabel("Energy per trial (J, log scale)")
    ax.set_title("Energy consumption by implementation\n"
                  "(torch/discretized: MAC/ACC estimate -- specksim: HOST CPU energy -- "
                  "speck: REAL on-chip energy)",
                  fontsize=10)

    for i, label in enumerate(labels):
        ax.annotate(label, xy=(x_pos[i], means[i]), xytext=(0, 8),
                    textcoords="offset points", ha='center', fontsize=8, color='dimgray')
    ax.margins(y=0.25)  # headroom so the annotations above don't crowd the title

    fig.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/energy_summary.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved energy summary figure to {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 14. Evaluation loop -- RMSE in physical velocity units against ground
#     truth, for each implementation, plus cross-implementation agreement.
#     Every list/dict/print statement below is built from active_impls
#     dynamically (set once, at the top of the script, from --models) --
#     nothing here dangles on an undefined pred_X/sc_X for an
#     implementation that wasn't selected to run.
# ---------------------------------------------------------------------------
N_COMPARE = 1
compare_indices = select_evenly_spaced_indices(len(test_dataset), N_COMPARE)
print(f"Comparing {' / '.join(active_impls)} over {len(compare_indices)} "
      f"trials, evenly spaced: {compare_indices}")

vs_gt_keys = [f"{impl}_vs_gt" for impl in active_impls]
# Cross-implementation agreement rows only make sense when BOTH sides were
# actually run -- e.g. --models torch speck (no discretized/specksim)
# shouldn't produce a torch_vs_discretized or speck_vs_specksim row at all.
cross_keys = []
if "torch" in active_impls and "discretized" in active_impls:
    cross_keys.append("torch_vs_discretized")
if "speck" in active_impls and "specksim" in active_impls:
    cross_keys.append("speck_vs_specksim")
loss_keys = vs_gt_keys + cross_keys
loss_samples = {k: [] for k in loss_keys}
all_counts = {impl: defaultdict(list) for impl in active_impls}
# Wall-clock seconds for the ENTIRE run_X_model() call, per trial, per
# implementation -- for run_speck_model() this includes its own internal
# wait_time sleeps, deliberately: that IS the real, actual latency of
# getting a prediction back from that implementation, not "compute time
# with the necessary waiting excluded". time.perf_counter() (not
# time.time()) since it's a monotonic clock meant specifically for
# measuring short intervals, unaffected by system clock adjustments.
latency_samples = {impl: [] for impl in active_impls}
# Energy: torch/discretized use the MAC/ACC-based op_energy_estimate.py
# estimate now (see run_torch_model()'s own docstring) -- NOT EnergyMeter,
# despite both running as ordinary Python on this machine; only specksim
# still reports HOST energy via EnergyMeter (same class as the rest of
# this project). speck reports REAL ON-CHIP energy (via the devkit's own
# PowerMonitor, see run_speck_model()). Three genuinely different kinds
# of number, not one ruler applied four times -- see module docstring
# point 7 and the printed ENERGY SUMMARY caveat below before comparing
# them across implementations.
# None values (from EnergyMeter's own RAPL counter wraparound mid-trial,
# for specksim specifically) are kept as None here too, not zeroed --
# filtered out at summary time.
energy_samples = {impl: [] for impl in active_impls}
energy_methods = {impl: None for impl in active_impls}
# Pooled (not per-trial) prediction/target accumulation, for cross-
# correlation -- CC over a handful of short, independent per-trial
# sequences isn't very meaningful; pooling every compared timestep across
# the WHOLE session first, then computing ONE cc_x/cc_y pair over that,
# matches both statistical convention and combined_metrics.json's own
# granularity (one cc_x/cc_y PER SESSION, not per trial) -- see
# _average_cc()'s own docstring in decoder_comparison_4x2.py, which this
# is built to merge directly into.
pred_phys_accum = {impl: [] for impl in active_impls}
target_phys_accum = []

for plot_i, trial_idx in enumerate(compare_indices):
    _, input_trial, target_trial = test_dataset[trial_idx]
    input_trial = input_trial.float()    # (T, C)
    target_trial = target_trial.float()  # (T, 2), SCALED space
    # (T, C) -> (T, C, 1, 1) -- matches DYNAPCNN_INPUT_SHAPE's (C, 1, 1)
    input_frames = input_trial.unsqueeze(-1).unsqueeze(-1)
    T = input_trial.shape[0]

    preds = {}
    spike_counts_by_impl = {}

    if "torch" in active_impls:
        (preds["torch"], spike_counts_by_impl["torch"], torch_latency_s,
         torch_energy_j_low, torch_energy_j_high, torch_energy_method) = run_torch_model(snn_seq, input_trial)
        latency_samples["torch"].append(torch_latency_s)
        energy_samples["torch"].append({"low": torch_energy_j_low, "high": torch_energy_j_high})
        energy_methods["torch"] = torch_energy_method

    if "discretized" in active_impls:
        (preds["discretized"], spike_counts_by_impl["discretized"], disc_latency_s,
         disc_energy_j_low, disc_energy_j_high, disc_energy_method) = run_torch_model(snn_disc, input_trial)
        latency_samples["discretized"].append(disc_latency_s)
        energy_samples["discretized"].append({"low": disc_energy_j_low, "high": disc_energy_j_high})
        energy_methods["discretized"] = disc_energy_method

    if "specksim" in active_impls:
        (preds["specksim"], spike_counts_by_impl["specksim"], specksim_latency_s,
         specksim_energy_j, specksim_energy_method) = run_specksim_model(
            specksim_snn, input_frames, n_spike_layers)
        latency_samples["specksim"].append(specksim_latency_s)
        energy_samples["specksim"].append(specksim_energy_j)
        energy_methods["specksim"] = specksim_energy_method

    if "speck" in active_impls:
        (preds["speck"], spike_counts_by_impl["speck"], speck_latency_s,
         speck_energy_j, speck_energy_method) = run_speck_model(input_frames, T)
        latency_samples["speck"].append(speck_latency_s)
        energy_samples["speck"].append(speck_energy_j)
        energy_methods["speck"] = speck_energy_method

    target_phys = unscale_velocity(target_trial, V_LO, V_HI, V_MARGIN)
    rmse_vs_gt = {}
    for impl in active_impls:
        pred_phys = unscale_velocity(preds[impl], V_LO, V_HI, V_MARGIN)
        rmse_vs_gt[impl] = torch.sqrt(criterion(pred_phys, target_phys))
        loss_samples[f"{impl}_vs_gt"].append(rmse_vs_gt[impl].item())
        pred_phys_accum[impl].append(pred_phys.numpy())
    target_phys_accum.append(target_phys.numpy())

    cross_str = ""
    if "torch_vs_discretized" in cross_keys:
        rmse_t_d = torch.sqrt(criterion(preds["torch"], preds["discretized"]))
        loss_samples["torch_vs_discretized"].append(rmse_t_d.item())
        cross_str += f" Torch<->Disc={rmse_t_d.item():.4f}"
    if "speck_vs_specksim" in cross_keys:
        rmse_speck_ss = torch.sqrt(criterion(preds["speck"], preds["specksim"]))
        loss_samples["speck_vs_specksim"].append(rmse_speck_ss.item())
        cross_str += f" Speck<->SpeckSim={rmse_speck_ss.item():.4f}"

    vs_gt_str = " ".join(f"{impl.capitalize()}={rmse_vs_gt[impl].item():.4f}" for impl in active_impls)
    latency_str = " ".join(f"{impl.capitalize()}={latency_samples[impl][-1]*1000:.1f}ms"
                            for impl in active_impls)
    print(f"\nSample {trial_idx}: {vs_gt_str}{cross_str}")
    print(f"           latency: {latency_str}")

    for impl in active_impls:
        for layer, counts in spike_counts_by_impl[impl].items():
            all_counts[impl][layer].append(counts)

    if PLOT_PREDICTIONS and plot_i < PLOT_MAX_SAMPLES:
        plot_idx = select_plot_indices(T)
        if len(plot_idx) < T:
            print(f"  [plot] trial {trial_idx} has {T} timesteps -- animating {len(plot_idx)} "
                  f"frames (every {PLOT_STRIDE}th), covering up to timestep {plot_idx[-1]} "
                  f"({(plot_idx[-1] + 1) / T * 100:.1f}% of the trial). Position/velocity are "
                  f"reconstructed from the FULL trial first, THEN subsampled for display -- "
                  f"see PLOT_STRIDE's own comment for why that order matters. The static "
                  f"velocity-over-time plot below still covers the FULL trial regardless.")

        pos_by_impl_px, pos_true_px = make_comparison_frames(
            {impl: preds[impl] for impl in active_impls}, target_trial.numpy())  # FULL trial
        pos_by_impl_px = {impl: px[plot_idx] for impl, px in pos_by_impl_px.items()}
        pos_true_px = pos_true_px[plot_idx]
        plot_comparison(trial_idx, pos_by_impl_px, pos_true_px)
        plot_crosshair_comparison(trial_idx, pos_by_impl_px, pos_true_px)

        # RAW velocity plots -- torch and speck specifically (the pair
        # requested), not all active_impls, though both new functions
        # work with any subset. Falls back to whatever of the two is
        # actually running if --models excluded one of them, rather
        # than assuming both are always present.
        velocity_impls = [i for i in ("torch", "speck") if i in active_impls]
        phys_by_impl = {impl: unscale_velocity(preds[impl], V_LO, V_HI, V_MARGIN).numpy()
                         for impl in velocity_impls}
        # plot_velocity_over_time() is a static fig.savefig() PNG, not a
        # FuncAnimation -- none of the pillow-buffering memory risk the
        # animations above have, so it deliberately gets the FULL,
        # un-strided, un-truncated trial, not plot_idx.
        plot_velocity_over_time(trial_idx, phys_by_impl, target_phys.numpy())

        # The velocity CROSSHAIR view below, unlike the plot just above,
        # IS a FuncAnimation -- velocity_to_pixel() is a plain elementwise
        # rescale (no cross-timestep integration, unlike position), so
        # stride order doesn't matter for correctness here the way it does
        # for position, but subsampling the OUTPUT keeps this consistent
        # with every other animated call site above.
        vel_px_by_impl = {impl: velocity_to_pixel(phys)[plot_idx] for impl, phys in phys_by_impl.items()}
        vel_px_true = velocity_to_pixel(target_phys.numpy())[plot_idx]
        plot_crosshair_comparison(trial_idx, vel_px_by_impl, vel_px_true, label="velocity")

print(f"\n{'='*72}\nRMSE SUMMARY OVER {len(compare_indices)} SAMPLES (physical velocity units "
      f"for vs.-ground-truth rows; SCALED-space RMSE for cross-implementation agreement rows)"
      f"\n{'='*72}")
labels = {"torch_vs_gt": "Torch vs GT", "discretized_vs_gt": "Discretized vs GT",
          "specksim_vs_gt": "SpeckSim vs GT", "speck_vs_gt": "Speck vs GT",
          "torch_vs_discretized": "Torch <-> Discretized",
          "speck_vs_specksim": "Speck <-> SpeckSim"}
for key in loss_keys:
    arr = np.array(loss_samples[key])
    print(f"  {labels[key]:<24}  median={np.median(arr):8.4f}  mean={arr.mean():8.4f}  std={arr.std():8.4f}")

if not RUN_ON_SPECK_HARDWARE:
    print("\n('speck' rows omitted -- not in --models. Once other implementations look "
          "right locally, add 'speck' to --models on the machine connected to the "
          "devkit to add the real chip to this comparison.)")

# --- Pooled cross-correlation (cc_x, cc_y), one pair PER IMPLEMENTATION,
# over every compared timestep across all N_COMPARE trials concatenated
# together -- see the accumulator's own comment before the evaluation
# loop for why pooled rather than per-trial. np.corrcoef, not
# bmi.metrics.pearson_corrcoef -- this file deliberately avoids that
# import chain (see snn_inference_utils.py's own module docstring for
# the same reasoning applied to load_snn_model()); mathematically
# identical for a plain Pearson correlation between two 1-D sequences,
# just computed directly rather than pulling in a heavier dependency for
# it. Matches combined_metrics.json's own cc_x/cc_y schema exactly (see
# decoder_comparison_4x2.py's _average_cc()), specifically so this file's
# own output can be merged into that same figure directly.
cc_vs_gt = {}
target_flat = np.concatenate(target_phys_accum, axis=0)  # (N_total, 2)
for impl in active_impls:
    pred_flat = np.concatenate(pred_phys_accum[impl], axis=0)  # (N_total, 2)
    cc_x = float(np.corrcoef(pred_flat[:, 0], target_flat[:, 0])[0, 1])
    cc_y = float(np.corrcoef(pred_flat[:, 1], target_flat[:, 1])[0, 1])
    cc_vs_gt[impl] = (cc_x, cc_y)

print(f"\n{'='*72}\nCROSS-CORRELATION (CC) SUMMARY OVER {len(compare_indices)} SAMPLES, "
      f"POOLED (physical velocity units)\n{'='*72}")
print(f"  {'Impl':<14}  {'CC_x':>8}  {'CC_y':>8}  {'Average CC':>10}")
for impl in active_impls:
    cc_x, cc_y = cc_vs_gt[impl]
    print(f"  {impl.capitalize():<14}  {cc_x:8.4f}  {cc_y:8.4f}  {(cc_x + cc_y) / 2:10.4f}")

# --- RMSE under BOTH aggregation conventions, per axis ---------------------------------------
# The two pipelines this project compares define their headline "RMSE" DIFFERENTLY, and the
# difference is not small. test_all_decoders.py (Oscar) uses sklearn's root_mean_squared_error
# with its default multioutput='uniform_average', which is the ARITHMETIC MEAN of the per-axis
# RMSEs: (rmse_x + rmse_y) / 2. This script's own rmse_vs_gt -- like train_bmi.py's best_loss --
# is torch.sqrt(MSELoss), i.e. the POOLED sqrt((mse_x + mse_y) / 2). By the QM-AM inequality
# pooled >= mean-of-axes, with a gap that grows the more unequal the two axes' errors are (for a
# real session: x=53.6, y=24.6 -> 39.10 vs 41.71, a 2.6-unit gap from definition alone, with
# IDENTICAL predictions). Verified against all five decoder rows in a real Oscar log (each
# headline equals the mean of its two printed per-axis values to 4 decimals). Reporting both, per
# axis, means torch/speck can be compared to Oscar's decoders like-for-like without guessing which
# convention a given number used; rmse_mean_of_axes is the Oscar-comparable one.
rmse_axes = {}
for impl in active_impls:
    pred_flat = np.concatenate(pred_phys_accum[impl], axis=0)  # (N_total, 2)
    _err = pred_flat - target_flat
    _rx = float(np.sqrt(np.mean(_err[:, 0] ** 2)))
    _ry = float(np.sqrt(np.mean(_err[:, 1] ** 2)))
    rmse_axes[impl] = {"rmse_x": _rx, "rmse_y": _ry,
                       "rmse_mean_of_axes": (_rx + _ry) / 2,   # == Oscar / sklearn 'uniform_average'
                       "rmse_pooled": float(np.sqrt(np.mean(_err ** 2)))}  # == train_bmi best_loss convention

print(f"\n{'='*72}\nRMSE BY AXIS, POOLED OVER {len(compare_indices)} SAMPLES (physical velocity units) -- "
      f"two conventions\n{'='*72}")
print(f"  {'Impl':<14}  {'RMSE_x':>8}  {'RMSE_y':>8}  {'mean of axes':>13}  {'pooled':>8}")
for impl in active_impls:
    _a = rmse_axes[impl]
    print(f"  {impl.capitalize():<14}  {_a['rmse_x']:8.4f}  {_a['rmse_y']:8.4f}  "
          f"{_a['rmse_mean_of_axes']:13.4f}  {_a['rmse_pooled']:8.4f}")
print("  'mean of axes' = Oscar/sklearn convention (compare against test_all_decoders.py's RMSE);")
print("  'pooled' = torch.sqrt(MSELoss), the convention of every RMSE printed ABOVE this table and of")
print("  train_bmi.py's best_loss.")

# --- Prediction "smoothness" -- does this implementation's own output
# track the ground truth's own frame-to-frame variability, stay too flat
# (resistant to change -- exactly the failure mode a near-constant
# prediction produces: RMSE can look fine if the ground truth spends
# most of its time near that constant value, while CC collapses because
# the prediction never actually tracks real variation), or overshoot it
# (too erratic/noisy)? Computed as std(diff(prediction)) / std(diff(
# ground truth)), POOLED across trials but diffed WITHIN each trial
# first (pred_phys_accum[impl] is a list of one (T,2) array per trial --
# diffing before concatenating avoids a spurious, large "jump" at every
# trial boundary that would otherwise inflate this number for reasons
# that have nothing to do with the implementation's own behavior). ~1.0
# means this implementation moves as much as the ground truth itself,
# frame to frame; <<1.0 is the "too static" signature; >>1.0 is "too
# erratic". x/y axes kept SEPARATE (not pre-averaged) since a real
# failure could plausibly affect one axis much more than the other.
print(f"\n{'='*72}\nPREDICTION SMOOTHNESS (frame-to-frame std, POOLED WITHIN each trial "
      f"first) -- ratio to ground truth's own\n{'='*72}")
gt_diff_x = np.concatenate([np.diff(t[:, 0]) for t in target_phys_accum])
gt_diff_y = np.concatenate([np.diff(t[:, 1]) for t in target_phys_accum])
gt_std_x, gt_std_y = float(gt_diff_x.std()), float(gt_diff_y.std())
print(f"  Ground truth itself: std(diff)_x={gt_std_x:.4f}, std(diff)_y={gt_std_y:.4f} "
      f"(physical velocity units / timestep)")
print(f"  {'Impl':<14}  {'std(diff)_x':>12}  {'std(diff)_y':>12}  "
      f"{'ratio_x':>8}  {'ratio_y':>8}")
smoothness = {}
for impl in active_impls:
    pred_diff_x = np.concatenate([np.diff(p[:, 0]) for p in pred_phys_accum[impl]])
    pred_diff_y = np.concatenate([np.diff(p[:, 1]) for p in pred_phys_accum[impl]])
    std_x, std_y = float(pred_diff_x.std()), float(pred_diff_y.std())
    ratio_x = std_x / gt_std_x if gt_std_x > 0 else float("nan")
    ratio_y = std_y / gt_std_y if gt_std_y > 0 else float("nan")
    smoothness[impl] = {"std_diff_x": std_x, "std_diff_y": std_y,
                         "ratio_to_gt_x": ratio_x, "ratio_to_gt_y": ratio_y}
    flag = ""
    if ratio_x < 0.5 or ratio_y < 0.5:
        flag = "  <-- notably FLATTER than ground truth (resistant to change)"
    elif ratio_x > 2.0 or ratio_y > 2.0:
        flag = "  <-- notably MORE ERRATIC than ground truth"
    print(f"  {impl.capitalize():<14}  {std_x:12.4f}  {std_y:12.4f}  "
          f"{ratio_x:8.3f}  {ratio_y:8.3f}{flag}")

# --- Spike density per layer, per implementation -- all_counts was
# already being accumulated (one entry per compared trial, each a list
# of that trial's own per-timestep spike counts for this layer) but
# never actually summarized or reported anywhere. mean_per_timestep is
# the layer's own average firing rate; silent_frac is the fraction of
# ALL compared timesteps where this layer fired ZERO spikes -- a direct,
# checkable signal for "is this layer effectively dead" (spike
# starvation) vs. "is this layer saturating" (silent_frac near 0 but
# mean_per_timestep unusually high compared to other implementations'
# own same layer).
print(f"\n{'='*72}\nSPIKE DENSITY PER LAYER, POOLED OVER {len(compare_indices)} SAMPLES\n{'='*72}")
spike_density = {impl: {} for impl in active_impls}
for impl in active_impls:
    print(f"  {impl.capitalize()}:")
    for layer in sorted(all_counts[impl].keys()):
        flat = np.concatenate([np.asarray(trial_counts) for trial_counts in all_counts[impl][layer]])
        mean_per_ts = float(flat.mean())
        silent_frac = float((flat == 0).mean())
        spike_density[impl][layer] = {"mean_per_timestep": mean_per_ts, "silent_frac": silent_frac}
        flag = "  <-- >90% silent, likely dead/starved" if silent_frac > 0.9 else ""
        print(f"    layer {layer}: mean={mean_per_ts:.3f} spikes/timestep, "
              f"silent={silent_frac*100:.1f}% of timesteps{flag}")

print(f"\n{'='*72}\nLATENCY SUMMARY OVER {len(compare_indices)} SAMPLES "
      f"(DECODE-LOOP-ONLY wall-clock seconds -- {T} timesteps, this project's fixed "
      f"windowed-mode trial length)"
      f"\n{'='*72}")
print(f"  {'Impl':<14}  {'Median (s)':>10}  {'Mean (s)':>10}  {'Std (s)':>10}  {'Per-timestep (ms)':>18}")
for impl in active_impls:
    arr = np.array(latency_samples[impl])
    per_timestep_ms = (arr.mean() / T) * 1000
    print(f"  {impl.capitalize():<14}  {np.median(arr):10.4f}  {arr.mean():10.4f}  "
          f"{arr.std():10.4f}  {per_timestep_ms:18.3f}")
print(f"\n  ('Per-timestep' divides mean total latency by {T} -- this "
      f"project's native step is 4ms, so per-timestep values above 4ms mean this "
      f"implementation currently runs SLOWER than real-time for continuous deployment. "
      f"DECODE-LOOP-ONLY means this EXCLUDES model/chip setup -- sinabs.reset_states()/"
      f"hook registration for torch/discretized, reset_states()/monitor setup for "
      f"specksim, and v_mem-reset-to-zero + 1s settle for speck -- none of that is "
      f"decoding, so none of it counts toward a number meant to answer 'how long does "
      f"decoding take.' Every implementation's energy_j above is measured over this "
      f"EXACT SAME window, for the same reason.)")

print(f"\n{'='*72}\nENERGY SUMMARY OVER {len(compare_indices)} SAMPLES "
      f"({T} timesteps/trial)\n{'='*72}")
print(f"  {'Impl':<14}  {'Mean (J/trial)':>24}  {'Per-timestep (mJ)':>24}  "
      f"{'Method':<20}  {'Implied cores':>13}")
implausible_impls = []
for impl in active_impls:
    raw_samples = [e for e in energy_samples[impl] if e is not None]
    n_null = len(energy_samples[impl]) - len(raw_samples)
    null_note = f" ({n_null} null skipped)" if n_null else ""
    method = energy_methods.get(impl) or "n/a"
    if len(raw_samples) == 0:
        print(f"  {impl.capitalize():<14}  no valid energy readings{null_note}")
        continue

    # torch/discretized now report a RANGE (dict {"low","high"}) from the
    # MAC/ACC estimator, not a single measured joules value -- see
    # run_torch_model()'s own docstring for why this is a range, not a
    # collapsed single number. specksim/speck are unaffected: still real,
    # single-valued measurements (EnergyMeter host energy / chip_power_monitor).
    is_range = isinstance(raw_samples[0], dict)
    if is_range:
        low_arr = np.array([e["low"] for e in raw_samples], dtype=float)
        high_arr = np.array([e["high"] for e in raw_samples], dtype=float)
        mean_str = f"[{low_arr.mean():.6f}, {high_arr.mean():.6f}]"
        per_timestep_low_mj = (low_arr.mean() / T) * 1000
        per_timestep_high_mj = (high_arr.mean() / T) * 1000
        per_timestep_str = f"[{per_timestep_low_mj:.4f}, {per_timestep_high_mj:.4f}]"
        implied_cores_str = "n/a"  # implied-cores is a proxy_psutil-specific
        # diagnostic (see below) -- the MAC/ACC estimate has no thread-count
        # proxy to reconstruct at all
        print(f"  {impl.capitalize():<14}  {mean_str:>24}  {per_timestep_str:>24}  "
              f"{method:<20}  {implied_cores_str:>13}{null_note}")
        continue

    arr = np.array(raw_samples, dtype=float)
    per_timestep_mj = (arr.mean() / T) * 1000
    # implied_cores: proxy_psutil computes energy_j = ASSUMED_CPU_TDP_WATTS *
    # cpu_frac * latency_s, so dividing back out reconstructs cpu_frac --
    # i.e. roughly how many logical cores torch/etc. were actually using
    # during that call. A number far above whatever ENERGY_METER_NUM_THREADS
    # (or your actual core budget) was set to means the psutil-based energy
    # estimate is inflated by uncontrolled multithreading, NOT that the
    # workload is somehow using more power than a modern CPU physically can
    # -- see energy_meter.py's own thread-pinning docstring for the full
    # mechanism. Not meaningful for 'rapl' or 'chip_power_monitor' rows
    # (real hardware counters, not derived from a thread-count proxy).
    implied_cores_str = "n/a"
    if method == "proxy_psutil":
        mean_latency_s = np.mean(latency_samples[impl])
        if mean_latency_s > 0:
            implied_cores = arr.mean() / (ASSUMED_CPU_TDP_WATTS * mean_latency_s)
            implied_cores_str = f"{implied_cores:.2f}"
            if implied_cores > 4:  # generous -- most laptops/workstations have <=8-16
                # logical cores, but even 4+ fully-loaded cores on ONE inference call
                # is already unusual enough to flag rather than silently trust
                implausible_impls.append((impl, implied_cores))
    print(f"  {impl.capitalize():<14}  {arr.mean():24.6f}  {per_timestep_mj:24.4f}  "
          f"{method:<20}  {implied_cores_str:>13}{null_note}")
if implausible_impls:
    flagged = ", ".join(f"{impl} (~{cores:.1f} cores)" for impl, cores in implausible_impls)
    print(f"\n  WARNING: {flagged} implies more CPU threads than this workload plausibly "
          f"needed for a single inference call -- the proxy_psutil energy number(s) above "
          f"are very likely INFLATED by uncontrolled multithreading (BLAS/torch spinning up "
          f"every core on this machine), not a real reflection of power draw. Set "
          f"ENERGY_METER_NUM_THREADS to your intended core count before re-running, or fix "
          f"RAPL permissions (see energy_meter.py) to get real joules instead of this proxy "
          f"entirely.")
print(f"\n  CAUTION -- these are up to THREE genuinely different kinds of number, not one "
      f"ruler applied several times (see module docstring point 7): torch/discretized "
      f"energy is a MAC/ACC-based ESTIMATE (method='mac_acc_estimate', see "
      f"op_energy_estimate.py) -- hardware-agnostic operation counts converted to joules "
      f"via reference per-op energy figures, reported as a [low, high] range (SRAM-only "
      f"vs. DRAM-only memory-access assumption), NOT a measurement of anything that "
      f"actually ran on this machine. specksim energy, when active, is still HOST energy "
      f"(this process's own CPU, via EnergyMeter -- 'rapl' = real hardware joules if this "
      f"node exposes them, 'proxy_psutil' = a rough elapsed_time x cpu_percent x "
      f"assumed-TDP estimate otherwise). speck energy is REAL ON-CHIP energy from the "
      f"devkit's own PowerMonitor telemetry (method='chip_power_monitor'), NOT an "
      f"estimate or a host-side proxy. Comparing an estimated computational cost against "
      f"real measured energy (host or chip) IS the comparison this deployment work exists "
      f"to make -- just don't mistake any of these for 'the same measurement, done more/"
      f"less precisely': they answer genuinely different questions ('how does this scale "
      f"algorithmically' vs. 'what did this specific run actually cost').")

if PLOT_PREDICTIONS:
    plot_energy_summary(energy_samples, energy_methods, active_impls)

# ---------------------------------------------------------------------------
# 14b. Save per-session metrics as JSON -- purely serializes the SAME
# loss_samples/latency_samples/energy_samples/energy_methods dicts the
# printed summaries above already use, computing nothing new. This is
# what a later aggregation step (across every session) reads -- see the
# accompanying driver script. Written into the SAME PLOT_SAVE_DIR as this
# session's own GIFs/plots, so everything for one session lives together.
# ---------------------------------------------------------------------------
if PLOT_SAVE_DIR is not None:
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    def _summarize_energy_samples(samples, method):
        """Handles BOTH shapes energy_samples[impl] can now take: a plain
        float per trial (specksim/speck -- real, single-valued measurements)
        or a {"low","high"} dict per trial (torch/discretized -- the
        MAC/ACC estimator's range, see run_torch_model()'s own docstring
        for why it's a range rather than one collapsed number)."""
        valid = [e for e in samples if e is not None]
        base = {"values": samples, "method": method or "n/a"}  # keep None
        # entries visible in "values", not silently dropped
        if not valid:
            return {**base, "mean_valid": None, "std_valid": None}
        if isinstance(valid[0], dict):
            lows = np.array([e["low"] for e in valid], dtype=float)
            highs = np.array([e["high"] for e in valid], dtype=float)
            return {**base,
                    "mean_valid_low": float(lows.mean()), "std_valid_low": float(lows.std()),
                    "mean_valid_high": float(highs.mean()), "std_valid_high": float(highs.std())}
        arr = np.array(valid, dtype=float)
        return {**base, "mean_valid": float(arr.mean()), "std_valid": float(arr.std())}

    metrics_out = {
        "session_id": SESSION_ID,
        "experiment": args.experiment,
        "subject": args.subject,
        "velocity_scale_checkpoint": VELOCITY_SCALE_CHECKPOINT,
        "velocity_scale_dataset": VELOCITY_SCALE_DATASET,
        "velocity_scaling_consistent": VELOCITY_SCALING_CONSISTENT,
        "checkpoint_provenance": CHECKPOINT_PROVENANCE,
        "checkpoint_path": CHECKPOINT_PATH,
        "dataset_path": SNN_DATASET_PATH,
        "active_impls": active_impls,
        "n_samples": len(compare_indices),
        "trial_length_timesteps": T,
        "dynapcnn_param_count": DYNAPCNN_PARAM_COUNT,  # None unless NEEDS_DYNAPCNN
        # ran (see that variable's own init comment) -- total weights+biases of
        # what actually gets deployed to the chip, not the pre-conversion model.
        "rmse": {
            key: {"values": loss_samples[key],
                  "median": float(np.median(loss_samples[key])),
                  "mean": float(np.mean(loss_samples[key])),
                  "std": float(np.std(loss_samples[key]))}
            for key in loss_keys
        },
        "rmse_axes": {f"{impl}_vs_gt": rmse_axes[impl] for impl in active_impls},
        "speck_input_audit": dict(SPECK_INPUT_AUDIT),
        "speck_wait_time_s": float(args.speck_wait_time) if "speck" in active_impls else None,
        "speck_raster_dt": float(args.speck_raster_dt) if "speck" in active_impls else None,
        "plots_generated": not NO_PLOTS,
        "power_audit": dict(POWER_AUDIT),
        "crosscheck_model_vs_snn_seq_max_abs_diff": float(_max_diff),
        "cc": {
            # Keyed "{impl}_vs_gt", matching "rmse"'s own key convention --
            # POOLED across all compared trials (see the accumulator's own
            # comment before the evaluation loop), one cc_x/cc_y pair per
            # implementation, not per trial. Matches combined_metrics.json's
            # own cc_x/cc_y schema exactly (see decoder_comparison_4x2.py's
            # _average_cc()) for direct merge compatibility.
            f"{impl}_vs_gt": {"cc_x": cc_vs_gt[impl][0], "cc_y": cc_vs_gt[impl][1],
                               "cc_avg": (cc_vs_gt[impl][0] + cc_vs_gt[impl][1]) / 2}
            for impl in active_impls
        },
        "prediction_smoothness": {
            # ratio_to_gt_x/y near 1.0 = tracks the ground truth's own
            # frame-to-frame variability; <<1.0 = too static/resistant to
            # change; >>1.0 = too erratic. See the print block above (same
            # computation) for the full reasoning.
            "ground_truth_std_diff": {"x": gt_std_x, "y": gt_std_y},
            **smoothness,
        },
        "spike_density": spike_density,  # {impl: {layer: {mean_per_timestep, silent_frac}}}
        "latency_s": {
            impl: {"values": latency_samples[impl],
                   "median": float(np.median(latency_samples[impl])),
                   "mean": float(np.mean(latency_samples[impl])),
                   "std": float(np.std(latency_samples[impl])),
                   "per_timestep_ms": float(np.mean(latency_samples[impl]) / T * 1000)}
            for impl in active_impls
        },
        "energy_j": {
            impl: _summarize_energy_samples(energy_samples[impl], energy_methods.get(impl))
            for impl in active_impls
        },
    }
    # Renamed from the old "metrics.json" -- "speck_results" more precisely
    # names what this file actually holds (RMSE/CC/latency/energy/param count
    # across torch/discretized/specksim/speck, one file per session), and
    # avoids the generic "metrics.json" name colliding, in a directory
    # listing, with test_all_decoders.py's own {session}_metrics.json.
    metrics_path = os.path.join(PLOT_SAVE_DIR, "speck_results.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved speck results to {metrics_path}")

# ---------------------------------------------------------------------------
# 15. Cleanup
# ---------------------------------------------------------------------------
if RUN_ON_SPECK_HARDWARE:
    samna_graph.stop()
    devkit.reset_board_soft(True)
    samna.device.close_device(devkit)
