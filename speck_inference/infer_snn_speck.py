"""
Minimal torch-vs-speck deployment comparison. Reports exactly three things, per
implementation, and nothing else:

    1. LOSS    -- RMSE (physical velocity units, x/y/pooled/mean-of-axes) and CC (x/y/avg)
                  vs ground truth.
    2. LATENCY -- per-SAMPLE processing time: the cost to decode ONE new 4ms window as it
                  arrives, with model state carried over from the previous window. NOT the
                  cost of processing a whole pre-loaded trial in one batched call -- a real
                  deployed decoder cannot batch samples it hasn't received yet, and this is
                  the SAME convention every ANN decoder in this project's efficiency
                  comparison already uses (see time_torch_per_timestep()'s own docstring for
                  why an earlier version of this script got this wrong for torch specifically
                  by measuring the batched-call number instead, and how it's now measured the
                  same way as everything else, including speck, which has no choice -- it
                  decodes physically one step at a time).
    3. POWER   -- REAL, MEASURED power draw, over that SAME per-sample measurement window.
                  Not an op-count estimate for either implementation:
                    torch : Intel RAPL (via energy_meter.EnergyMeter), wrapping the SAME
                            per-timestep decode loop LATENCY is measured on -- the actual CPU
                            package energy this machine spends per sample decoded, not per
                            whole-trial batch.
                    speck : the Speck2f devkit's own PowerMonitor telemetry, mean power over
                            the decode loop x the loop's measured duration (see
                            measure_chip_power()'s docstring for why NOT the events'
                            timestamps -- that was a real, ~1000x bug in an earlier version).

Deliberately stripped down from a much larger version of this script that also ran a
"discretized" (quantized, pre-chip) software twin and a "specksim" CPU-simulated chip, produced
GIFs/plots, and reported an extensive set of diagnostics (checkpoint provenance, spike density,
prediction smoothness, per-session input-event and power-timestamp audits). None of that features
here. What's kept, verbatim or near-verbatim, is kept because it was a real, hard-won correctness
fix and dropping it would silently reintroduce a bug already paid for once:

  - tau_syn=None is hard-coded in snn_inference_utils.load_snn_model() (this project's
    checkpoints were trained without synaptic dynamics, and the chip cannot realize tau_syn
    regardless -- see that file's own comment). Nothing here overrides it, so there is no
    --override-tau-syn flag any more: the earlier need for it was diagnostic, not a real setting
    a run should ever change.
  - The EMA CASCADE (ema_cascade_update / decode_from_ema) -- needed because the chip only
    returns raw per-timestep spikes, and this project's model applies temporal_decay_stages of
    EMA filtering plus population decoding on top of that, external to the chip. Get this wrong
    and the chip's decode is silently wrong: this was the single largest bug found in this
    script's history (predictions decoded through a 1-stage filter instead of the trained
    model's own 2-stage cascade, worth several RMSE units).
  - A one-time cross-check, run once before anything is deployed to the chip, that the flat
    nn.Sequential built for that deployment (snn_seq) actually reproduces the trained model's own
    forward() -- catches exactly the class of bug above before it reaches real hardware.
  - The RESET-SUPPRESSION technique in time_torch_per_timestep()/energy_torch_per_timestep(),
    reused (not reinvented) from plot_decoder_efficiency.py's own time_snn_per_timestep() --
    model_bmi.py's forward() calls sinabs.utils.reset_states() UNCONDITIONALLY every call, so
    measuring per-sample latency/energy with separate Python-level calls needs that suppressed
    for every timestep but a trial's first, or the sinabs neuron layers' own state never
    persists between calls at all.

Usage:
    python infer_snn_speck.py --experiment bmi --subject indy \
        --checkpoint-path ./checkpoints/bmi/mua/indy_20160407_02/best_model_weights.pth \
        --dataset-path ./datasets/bmi/mua/indy_20160407_02 \
        --output-dir ./speck_results/bmi/indy/indy_20160407_02 \
        --models torch speck
"""

import argparse
import json
import os
import time
from collections import defaultdict

# Single thread, deliberately, BEFORE energy_meter is imported (it reads this env var at ITS
# OWN import time) and before torch/numpy below -- torch/BLAS freely using every core is exactly
# what would inflate the RAPL reading with work this process didn't ask for. setdefault, not a
# hard overwrite, so an explicit `export ENERGY_METER_NUM_THREADS=N` still wins.
os.environ.setdefault("ENERGY_METER_NUM_THREADS", "1")

from energy_meter import EnergyMeter, pin_torch_threads  # noqa: E402 -- see the thread-pinning
# note above for why this import (and pin_torch_threads() right after `import torch` below)
# must come before torch/numpy are imported or used anywhere in this process.

import numpy as np
import torch
pin_torch_threads()
import torch.nn as nn
from tqdm import tqdm

from models.model_bmi import _SinabsNeuronLayer, _SJNeuronLayer
from snn_inference_utils import load_snn_model

_MODEL_ORDER = ["torch", "speck"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--experiment", type=str, default="bmi", choices=["bmi", "hkm"],
                    help="Which model module to load the checkpoint through.")
parser.add_argument("--subject", type=str, default=None,
                    help="Subject whose velocity_scalers.json to use (e.g. 'indy', 'loco'). "
                         "Pass explicitly on a dataset layout with no subject directory level.")
parser.add_argument("--models", nargs="+", default=["torch", "speck"], choices=_MODEL_ORDER,
                    help="Which implementations to run (default: both). 'speck' requires the "
                         "physical Speck2f devkit connected on this machine.")
parser.add_argument("--session-id", type=str, default=None,
                    help="Used to build default --checkpoint-path/--dataset-path if not given.")
parser.add_argument("--checkpoint-path", type=str, default=None,
                    help="Path to this session's best_model_weights.pth.")
parser.add_argument("--dataset-path", type=str, default=None,
                    help="Path to this session's test data directory.")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Where to save this session's results.json. Required to save anything.")
parser.add_argument("--speck-wait-time", type=float, default=0.001,
                    help="Real wall-clock seconds run_speck_model() waits after sending one "
                         "timestep's input before reading output spikes. Default 1ms.")
parser.add_argument("--speck-raster-dt", type=float, default=0.1,
                    help="dt passed to chip_factory.raster_to_events() per timestep.")
parser.add_argument("--torch-timing-warmup", type=int, default=1,
                    help="Untimed passes over test_dataset[0], per-timestep, before "
                         "measuring torch's per-sample LATENCY. Small default: this "
                         "project's trials (15,000-40,000+ timesteps) are already far "
                         "longer than plot_decoder_efficiency.py's own ~50-timestep "
                         "--n_timing_samples default needs 10 repeats for, so 1 warmup "
                         "pass is normally enough to settle any first-call overhead.")
parser.add_argument("--torch-timing-repeats", type=int, default=2,
                    help="Timed passes (median taken) for torch's per-sample LATENCY.")
parser.add_argument("--torch-energy-repeats", type=int, default=1,
                    help="Passes for torch's per-sample ENERGY, inside ONE EnergyMeter "
                         "block (see energy_torch_per_timestep()'s own docstring for why "
                         "this default is 1, not plot_decoder_efficiency.py's default of "
                         "10 -- that default compensates for a much SHORTER default "
                         "sample there).")
args = parser.parse_args()

active_impls = [m for m in _MODEL_ORDER if m in args.models]
RUN_ON_SPECK_HARDWARE = "speck" in active_impls

if args.session_id:
    SESSION_ID = args.session_id
elif args.dataset_path is not None:
    SESSION_ID = os.path.basename(os.path.normpath(args.dataset_path))
elif args.checkpoint_path is not None:
    SESSION_ID = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint_path)))
else:
    SESSION_ID = "indy_20160407_02"

CHECKPOINT_PATH = args.checkpoint_path or f"./checkpoints/bmi/mua/{SESSION_ID}/best_model_weights.pth"
DATASET_PATH = args.dataset_path or f"./datasets/bmi/mua/{SESSION_ID}"
BATCH_SIZE = 1

print(f"Session   : {SESSION_ID}")
print(f"Models    : {active_impls}")

# ---------------------------------------------------------------------------
# 1. Load checkpoint -- tau_syn=None is hard-coded inside load_snn_model()/
#    snn_inference_utils.py itself; nothing here needs to touch it.
# ---------------------------------------------------------------------------
model, checkpoint, velocity_scale = load_snn_model(CHECKPOINT_PATH, args.experiment)
model.eval()
V_LO, V_HI, V_MARGIN = velocity_scale
N_BINS = model.n_bins
POSITIONS = model.positions.clone()
NEUTRAL_SCALED_PRED = model.neutral_scaled_pred.clone()
TEMPORAL_DECAY = model.temporal_decay
TEMPORAL_DECAY_STAGES = model.temporal_decay_stages
print(f"Temporal decay: {TEMPORAL_DECAY:.4f}, {TEMPORAL_DECAY_STAGES} stage(s) "
      f"(EMA cascade depth)")


def unscale_velocity(v_scaled, lo, hi, margin):
    """Inverse of the dataloader's forward scaling. Deliberately NOT imported from
    train_bmi.py -- that module pulls in heavy, training-only dependencies an inference
    script has no business depending on transitively for one small, pure formula."""
    return lo + (hi - lo) * (v_scaled - margin) / (1 - 2 * margin)


# ---------------------------------------------------------------------------
# 2. Dataloader
# ---------------------------------------------------------------------------
from datasets.dataset import create_dataloaders

try:
    train_loader, test_loader = create_dataloaders(
        data_path=DATASET_PATH, batch_size=BATCH_SIZE, num_workers=0,
        shuffle_train=False, small=False, experiment=args.experiment, subject=args.subject)
except ValueError as exc:
    if args.subject is None and "Could not derive (experiment, subject)" in str(exc):
        raise ValueError(f"{exc}\n\n  -> pass --subject <subject> (e.g. --subject indy).") from exc
    raise
test_dataset = test_loader.dataset
print(f"Test dataset: {len(test_dataset)} trial(s) at {DATASET_PATH!r}")

# --- Velocity-scaling consistency check -- a real, silent-failure-prone mismatch: the
# dataset scales targets with velocity_scalers.json's per-subject bounds, while V_LO/V_HI/
# V_MARGIN above come from the checkpoint's own stored train_args. If they disagree,
# physical-unit RMSE below is silently distorted (CC is not: Pearson r is affine-invariant).
_dataset_scale = (getattr(test_dataset, "v_lo", None), getattr(test_dataset, "v_hi", None),
                  getattr(test_dataset, "v_margin", None))
if None not in _dataset_scale and not np.allclose((V_LO, V_HI, V_MARGIN), _dataset_scale):
    print("\n" + "!" * 72)
    print(f"  WARNING: velocity-scaling MISMATCH -- checkpoint lo={V_LO}, hi={V_HI}, "
          f"margin={V_MARGIN} vs dataset lo={_dataset_scale[0]}, hi={_dataset_scale[1]}, "
          f"margin={_dataset_scale[2]}. Physical-unit RMSE below is distorted.")
    print("!" * 72 + "\n")

# ---------------------------------------------------------------------------
# 3. The EMA cascade -- needed to decode the chip's raw per-timestep spikes exactly the way
#    the trained model itself does internally (model_bmi.py's own forward()). See this
#    file's own module docstring for why this specific piece is kept verbatim.
# ---------------------------------------------------------------------------


def decode_from_ema(ema: torch.Tensor) -> torch.Tensor:
    """(1, N_BINS*2) EMA-smoothed state -> (1, 2) (x, y) in SCALED space -- reimplements
    SNN_Speck.decode_output() exactly, including the zero-spike neutral_scaled_pred
    fallback. Required outside the model because the chip can only execute the raw neuron
    layers themselves, never this project's Python-side decode logic."""
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
    """One timestep of SNN_Speck.forward()'s temporal EMA CASCADE: stage 0 is fed this
    timestep's raw output-layer activity, each later stage is fed the PREVIOUS stage's own
    EMA output, and decoding reads the LAST stage. With stages=1 this is exactly one update.
    Getting this wrong (a single-stage EMA against a checkpoint trained with stages>=2) was
    this script's single largest historical bug -- worth several RMSE units."""
    stage_input = x
    for i in range(len(ema_stages)):
        ema_stages[i] = decay * ema_stages[i] + stage_input
        stage_input = ema_stages[i]
    return ema_stages[-1]


# ---------------------------------------------------------------------------
# 4. torch latency/energy: ONE quantity, applied identically to every decoder and every
#    implementation this project reports, no matter where it runs -- the cost to decode ONE
#    new 4ms sample AS IT ARRIVES, with model state carried over from the previous sample.
#    NOT the cost of processing a whole pre-loaded trial in one batched call: that's a
#    materially different, throughput-oriented quantity (confirmed directly: batched-call
#    torch measured ~0.47 ms/step here, vs ~1.5-1.6 ms/step for EVERY per-sample measurement
#    in this project -- KF/WF/LSTM/QRNN's own time_predict_per_sample(), Oscar's own SNN via
#    time_snn_per_timestep(), and the chip, which has no choice, it decodes physically one
#    step at a time). A real deployed decoder cannot batch samples that haven't arrived yet,
#    so per-sample is the only one of these two numbers that answers a real deployment
#    question -- batched throughput was the WRONG thing to report as "torch latency", even
#    though it was still the right, simplest way to get RMSE/CC (kept exactly as-is below,
#    untouched by this).
#
# time_torch_per_timestep() below is deliberately NOT a new technique -- it reuses
# plot_decoder_efficiency.py's own time_snn_per_timestep()/run_pass() reset-suppression
# pattern verbatim in spirit (same reason it exists there: model_bmi.py's own
# SNN_Speck.forward() calls sinabs.utils.reset_states(self) UNCONDITIONALLY at the top of
# EVERY call, so a per-timestep external Python loop needs that suppressed for every
# timestep except the trial's own first one, or the sinabs neuron layers' own v_mem/i_syn
# state never persists across the per-timestep calls at all).
#
# ONE THING THIS DOES NOT DO, ON PURPOSE: use this loop's own per-call OUTPUT for anything.
# forward()'s temporal_ema_stages (the EMA accumulator this project's decode depends on) is a
# PLAIN LOCAL VARIABLE inside forward() -- freshly reinitialized to zero at the top of every
# call, completely separate from sinabs' own v_mem/i_syn buffers, and NOT covered by the
# reset-suppression trick above. A per-timestep call sequence therefore times the real compute
# cost of one decode faithfully (the same work runs either way), but each call's own DECODED
# VALUE is meaningless -- there has been no time for the EMA to accumulate anything. Confirmed
# directly against model_bmi.py's own forward(): `temporal_ema_stages = [torch.zeros(...) ...]`
# sits INSIDE forward(), not on self. Exactly why plot_decoder_efficiency.py's own run_pass()
# never captures or uses snn_model(x_t)'s return value either -- this is a pure timing/energy
# harness on both sides, and RMSE/CC keep coming ONLY from the whole-trial model() call below,
# which is unaffected by any of this.
def time_torch_per_timestep(input_trial_t_n_c, n_warmup=1, n_repeats=2):
    """input_trial_t_n_c: (T, 1, C) -- ONE trial, batch size 1 (matches model()'s own expected
    shape). Returns latency_s: MEDIAN over n_repeats full passes (after n_warmup untimed
    passes) of "one pass over every timestep in this trial", divided by T -- i.e. seconds per
    SAMPLE, the same per-sample convention time_snn_per_timestep() reports latency_s in.
    """
    import sinabs
    real_reset_states = sinabs.utils.reset_states
    T = input_trial_t_n_c.shape[0]

    def run_pass():
        try:
            with torch.no_grad():
                for t in range(T):
                    sinabs.utils.reset_states = real_reset_states if t == 0 else (lambda *a, **k: None)
                    model(input_trial_t_n_c[t:t + 1])  # output deliberately discarded -- see above
        finally:
            sinabs.utils.reset_states = real_reset_states

    for _ in range(n_warmup):
        run_pass()
    pass_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        run_pass()
        pass_times.append(time.perf_counter() - t0)
    # Printed RAW, not just the divided-by-T number that gets reported downstream -- direct,
    # checkable evidence that this actually ran T SEPARATE per-timestep Python calls, not a
    # single batched one. A model whose forward() already loops per-timestep INTERNALLY (this
    # project's own model_bmi.py does: `for ts in range(T): ...`) can show a per-sample number
    # that looks numerically close to an old batched-call measurement even though the
    # underlying loop is now genuinely per-sample -- because the dominant cost either way is
    # that SAME internal T-iteration loop; wrapping it in 1 outer call vs T outer calls adds
    # only microseconds of extra Python dispatch on top, not milliseconds. This print is the
    # difference between "trust that arithmetic" and "see it directly": T x this reported
    # per-pass time is the SAME order of magnitude as an old batched call's own total wall
    # time for a trial this length would have been, and that is expected, not a sign this
    # code silently reverted to batching.
    print(f"    [torch timing] {n_warmup} warmup + {n_repeats} timed pass(es) over {T} timesteps, "
          f"each pass = {T} separate model() calls: pass times {[f'{p:.2f}s' for p in pass_times]}, "
          f"median {np.median(pass_times):.2f}s")
    return float(np.median(pass_times)) / T


def energy_torch_per_timestep(input_trial_t_n_c, n_energy_repeats=1):
    """Same reset-suppressed per-timestep loop as time_torch_per_timestep(), wrapped in
    EnergyMeter instead of perf_counter -- kept as a SEPARATE function/pass, not fused into
    the same loop, for the identical reason plot_decoder_efficiency.py's own
    time_snn_per_timestep() keeps them separate: RAPL's counters need a long-enough wall-clock
    window to be trustworthy, and mixing a `with EnergyMeter()` block into the ALREADY-timed
    latency loop would make the latency numbers include EnergyMeter's own (small but real)
    overhead. n_energy_repeats defaults to 1, not plot_decoder_efficiency.py's default of 10 --
    that default was calibrated for its OWN --n_timing_samples default of 50 timesteps, far too
    short a window on its own; this project's trials run 15,000-40,000+ timesteps, already
    comfortably longer than RAPL needs (single-digit milliseconds) even for one pass. Returns
    (energy_j_per_sample, energy_method, latency_s_of_this_pass) -- energy_j_per_sample is
    already divided by T*n_energy_repeats, matching every other per-sample energy figure in
    this project; latency_s_of_this_pass is returned too so the "too-short a window" caution
    below can check it directly rather than re-deriving it.
    """
    import sinabs
    real_reset_states = sinabs.utils.reset_states
    T = input_trial_t_n_c.shape[0]

    def run_pass():
        try:
            with torch.no_grad():
                for t in range(T):
                    sinabs.utils.reset_states = real_reset_states if t == 0 else (lambda *a, **k: None)
                    model(input_trial_t_n_c[t:t + 1])
        finally:
            sinabs.utils.reset_states = real_reset_states

    with EnergyMeter() as em:
        for _ in range(n_energy_repeats):
            run_pass()
    print(f"    [torch energy] {n_energy_repeats} pass(es) over {T} timesteps each -- "
          f"total window {em.latency_s:.2f}s ({T * n_energy_repeats} model() calls)")
    energy_j = None if em.energy_j is None else em.energy_j / (n_energy_repeats * T)
    if em.latency_s < 0.01:
        print(f"    [caution] torch energy measurement window was only {em.latency_s * 1000:.2f} ms "
              f"-- may be too short for a reliable {em.energy_method} reading; consider raising "
              f"--torch-energy-repeats")
    return energy_j, em.energy_method, em.latency_s


criterion = torch.nn.MSELoss()
pred_phys_accum = {impl: [] for impl in active_impls}
target_phys_accum = []
latency_s_total = {impl: 0.0 for impl in active_impls}
# The timestep count latency_s_total[impl] should be divided by, if it differs from n_steps
# (the RMSE pass's own timestep count) -- see the RESULTS block below. None means "same as
# n_steps", true for every impl EXCEPT torch after the per-sample timing redesign: torch's RMSE
# pass covers every test trial (T_total, via test_loader), but its latency/energy pass covers
# ONLY test_dataset[0] (T_timing) -- the SAME single trial speck decodes -- which is a
# DIFFERENT count whenever more than one test trial exists. speck's own RMSE and latency have
# always both been scoped to test_dataset[0] only, so n_steps already matches there by
# construction; only torch needs the override.
latency_n_steps = {impl: None for impl in active_impls}
energy_j_total = {impl: 0.0 for impl in active_impls}
energy_method = {impl: None for impl in active_impls}
T_total = 0
_sys_cpu_pct_samples = []  # whole-SYSTEM (not just this process) CPU utilization during each
# torch EnergyMeter window -- see the plausibility check printed after the loop for why.

if "torch" in active_impls:
    import psutil  # already a hard dependency of energy_meter.py itself -- importing it
    # directly here too costs nothing extra.
    psutil.cpu_percent(interval=None)  # prime -- the first call after process start is a
    # meaningless baseline read; this establishes the reference point the FIRST real
    # measurement's own reading is measured against.

    # --- RMSE/CC: the whole-trial batched call, UNCHANGED, UNTOUCHED -- already matches the
    # checkpoint's own best_loss exactly, and nothing about the latency/energy redesign below
    # has any bearing on this. No EnergyMeter here any more: its latency/energy were never
    # reported as the headline numbers now anyway (see the module-level comment above this
    # block), so there is no reason to keep measuring them on this call at all.
    with torch.no_grad():
        for labels, inputs, targets in test_loader:
            inputs = inputs.transpose(0, 1)    # (N,T,C) -> (T,N,C)
            targets = targets.transpose(0, 1)  # (N,T,2) -> (T,N,2)
            predictions, _, _ = model(inputs)
            T_total += inputs.shape[0]
            pred_phys_accum["torch"].append(unscale_velocity(predictions, V_LO, V_HI, V_MARGIN).reshape(-1, 2).numpy())
            target_phys_accum.append(unscale_velocity(targets, V_LO, V_HI, V_MARGIN).reshape(-1, 2).numpy())

    # --- LATENCY/ENERGY: the per-sample measurement, on the SAME single trial (test_dataset[0])
    # speck itself decodes -- see time_torch_per_timestep()/energy_torch_per_timestep()'s own
    # docstrings for the full reasoning. This, not the batched call above, is what gets reported.
    _, _torch_timing_input, _ = test_dataset[0]
    _torch_timing_input = _torch_timing_input.float().unsqueeze(1)  # (T,C) -> (T,1,C)
    latency_s_total["torch"] = time_torch_per_timestep(
        _torch_timing_input, n_warmup=args.torch_timing_warmup, n_repeats=args.torch_timing_repeats)
    _energy_j_per_sample, _energy_method, _energy_pass_latency_s = energy_torch_per_timestep(
        _torch_timing_input, n_energy_repeats=args.torch_energy_repeats)
    energy_method["torch"] = _energy_method
    # energy_j_total["torch"] is used downstream as a TOTAL (divided back out by latency_s_total
    # to get mW) -- see the results block further down. Reconstructed here as
    # per-sample-energy x T so that downstream arithmetic (unchanged from before this redesign)
    # keeps working without also having to change: energy_j_total["torch"] / latency_s_total["torch"]
    # still gives the correct mean power in Watts either way, since both are now per-sample x T.
    T_timing = _torch_timing_input.shape[0]
    latency_n_steps["torch"] = T_timing
    energy_j_total["torch"] = None if _energy_j_per_sample is None else _energy_j_per_sample * T_timing
    latency_s_total["torch"] = latency_s_total["torch"] * T_timing  # per-sample -> total-over-T_timing,
    # matching the SAME convention latency_s_total already used for every OTHER impl (a total
    # over some number of timesteps, divided by that count later to report ms/step) -- T_timing
    # here, NOT T_total (T_total is the whole-trial RMSE pass's own timestep count, a materially
    # different loop this measurement never touches).
    # WHOLE-SYSTEM (every process, every core) CPU utilization during the ENERGY pass specifically
    # -- see the plausibility check below for why this diagnostic exists at all.
    _sys_cpu_pct_samples.append(psutil.cpu_percent(interval=None))

    if energy_j_total["torch"] is None:
        print(f"  WARNING: RAPL energy unavailable (counter wraparound) -- torch power for this "
              f"session is unmeasurable.")

    # energy_method == 'proxy_psutil' means RAPL was never available on this machine at all --
    # a rough elapsed-time x cpu-percent x assumed-TDP GUESS, not a measurement. Every real run
    # of this pipeline so far has used 'rapl' (confirmed in every session log's own startup
    # line: "Energy measurement: using Intel RAPL ..."), so this is a defensive check for a
    # machine/permissions change, not an expected path -- but "concrete power consumption" is
    # this script's whole point, so a silent fallback to an estimate here would be exactly the
    # failure mode this rewrite exists to avoid, and must not pass quietly.
    if energy_method["torch"] == "proxy_psutil":
        print("  WARNING: RAPL is NOT available on this machine -- torch power below is a rough "
              "psutil-based ESTIMATE (elapsed_time x cpu_percent x an assumed 65W TDP), not a "
              "real measurement. Fix RAPL access (permissions on /sys/class/powercap/intel-rapl) "
              "before treating this number as 'concrete power'.")
    # PLAUSIBILITY CONTEXT: RAPL's package domain sums draw from EVERY process on the chip, not
    # just this one -- printed UNCONDITIONALLY (not gated behind a threshold) because whether a
    # given wattage reading is "plausible" for a laptop is inherently ambiguous (a gaming/
    # workstation laptop's package TDP can legitimately exceed 90W; a thin-and-light's idle
    # alone can be a large fraction of it), while system-wide CPU utilization during the SAME
    # window is the direct, causal signal for whether OTHER activity contributed to this
    # reading, regardless of what a wattage-only cutoff would say. A first version of this
    # check used a fixed watts-only ceiling (90W) chosen independently of any real
    # measurement; it did not fire on the actual case that motivated adding it (71.3W, i.e.
    # below that ceiling) -- kept as a cautionary note in this comment, not repeated as logic,
    # since a threshold picked without checking it against the motivating case is exactly the
    # kind of unverified guess this whole project has been about eliminating.
    _mean_sys_pct = sum(_sys_cpu_pct_samples) / len(_sys_cpu_pct_samples) if _sys_cpu_pct_samples else None
    _torch_watts = (energy_j_total["torch"] / latency_s_total["torch"]
                    if energy_j_total["torch"] is not None and latency_s_total["torch"] > 0 else None)
    if _torch_watts is not None:
        print(f"\n[torch power context] {_torch_watts:.2f} W measured, "
              f"{'mean system-wide CPU utilization during measurement: ' + format(_mean_sys_pct, '.1f') + '%' if _mean_sys_pct is not None else 'system-wide CPU utilization unavailable'} "
              f"(this process was pinned to 1 thread; RAPL sums the WHOLE package, every process, "
              f"so utilization elsewhere on the machine during this window inflates this reading "
              f"without being this trial's own cost)")
        # A single thread doing this architecture's ~1M ops/s should cost, at most, a few watts
        # over idle on any real CPU -- 10W is already a generous allowance for that. Above it,
        # OR alongside non-trivial system-wide load while pinned to one thread, the reading
        # should not be reported as this model's own power cost without first checking what
        # else was running.
        _watts_high = _torch_watts > 10.0
        _load_high = _mean_sys_pct is not None and _mean_sys_pct > 25.0
        if _watts_high or _load_high:
            print(f"  WARNING: this reading is likely NOT primarily this trial's own draw. "
                  f"{'The wattage alone is far above what one pinned thread doing this little compute should cost. ' if _watts_high else ''}"
                  f"{'System-wide CPU utilization was substantial for a run pinned to one thread. ' if _load_high else ''}"
                  f"Close other applications and re-run, or measure on an exclusively-allocated node "
                  f"instead (e.g. Oscar) before reporting this as the model's power cost.")
    _pw_str = (f"{energy_j_total['torch'] / latency_s_total['torch'] * 1e3:.4f} mW"
              if energy_j_total["torch"] is not None else "UNAVAILABLE (see warning above)")
    print(f"\n[torch] per-sample latency {latency_s_total['torch'] * 1000 / T_timing:.4f} ms/step "
          f"(n={T_timing} timesteps in trial 0, same trial speck decodes), power {_pw_str} "
          f"({energy_method['torch']}) -- RMSE/CC above are from the SEPARATE, unaffected "
          f"whole-trial pass over all {T_total} timestep(s) across every test trial.")

# ---------------------------------------------------------------------------
# 5. speck: build the flat nn.Sequential deployment target, cross-check it against the
#    trained model, then (if requested) convert and run it on the real chip.
# ---------------------------------------------------------------------------
NEEDS_DYNAPCNN = "speck" in active_impls
DYNAPCNN_PARAM_COUNT = None

if NEEDS_DYNAPCNN:
    if checkpoint.get("args", {}).get("neuron_type") != "iaf":
        raise RuntimeError(
            f"This checkpoint was trained with neuron_type="
            f"{checkpoint.get('args', {}).get('neuron_type')!r}. Only IAF is deployable to "
            f"the chip -- this session would need retraining with --neuron-type iaf.")

    seq_layers = []
    for m in model.layers:
        if isinstance(m, (_SinabsNeuronLayer, _SJNeuronLayer)):
            seq_layers.append(m.neuron)
        else:
            seq_layers.append(m)
    snn_seq = nn.Sequential(*seq_layers)

    # --- one-time cross-check: does snn_seq, run through the SAME EMA cascade, reproduce
    # model()'s own predictions on one real trial? If this fails, nothing built from snn_seq
    # below (dynapcnn_net, the physical chip config) should be trusted.
    import sinabs
    _, _first_input, _ = test_dataset[0]
    _first_input = _first_input.float()
    sinabs.reset_states(snn_seq)
    snn_seq.eval()
    _ema_stages = [torch.zeros(1, N_BINS * 2) for _ in range(TEMPORAL_DECAY_STAGES)]
    _seq_preds = []
    with torch.no_grad():
        for t in range(_first_input.shape[0]):
            x = snn_seq(_first_input[t].unsqueeze(0))
            _ema = ema_cascade_update(_ema_stages, x, TEMPORAL_DECAY)
            _seq_preds.append(decode_from_ema(_ema).squeeze(0))
        _seq_preds = torch.stack(_seq_preds, dim=0)
        _model_preds, _, _ = model(_first_input.unsqueeze(1))  # (T,C) -> (T,1,C)
    sinabs.reset_states(snn_seq)
    _max_diff = float((_seq_preds - _model_preds.squeeze(1)).abs().max())
    if _max_diff > 1e-4:
        raise RuntimeError(
            f"CROSS-CHECK FAILED: max |model() - snn_seq()| = {_max_diff:.6f} (scaled velocity "
            f"units), expected ~0. The flat Sequential built for chip deployment does NOT "
            f"reproduce the trained model -- do not proceed to DynapcnnNetwork conversion or "
            f"deploy to hardware until this is understood.")
    print(f"[cross-check] max |model() - snn_seq()| = {_max_diff:.2e} -- OK")

    from sinabs.backend.dynapcnn import DynapcnnNetwork
    DYNAPCNN_INPUT_SHAPE = (int(model.layers[0].in_features), 1, 1)
    snn_seq_for_dynapcnn = nn.Sequential(nn.Flatten(), *snn_seq)
    dynapcnn_net = DynapcnnNetwork(snn=snn_seq_for_dynapcnn, input_shape=DYNAPCNN_INPUT_SHAPE,
                                   discretize=True, dvs_input=False)
    DYNAPCNN_PARAM_COUNT = 0
    for layer in dynapcnn_net.sequence:
        for attr in ("weight", "bias"):
            try:
                DYNAPCNN_PARAM_COUNT += getattr(layer.conv_layer, attr).numel()
            except AttributeError:
                pass
    print(f"DynapcnnNetwork: {DYNAPCNN_PARAM_COUNT:,} parameters as deployed")

if RUN_ON_SPECK_HARDWARE:
    import samna
    import sinabs.backend.dynapcnn.io as sio
    from sinabs.backend.dynapcnn.chip_factory import ChipFactory

    DEVKIT_NAME = "speck2fdevkit:0"
    devkit = sio.open_device(DEVKIT_NAME)
    chip_factory = ChipFactory("speck2fdevkit")

    stop_watch = devkit.get_stop_watch()
    power_monitor = devkit.get_power_monitor()
    power_source_node = power_monitor.get_source_node()
    power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()
    POWER_SAMPLE_RATE_HZ = 100

    samna_graph = samna.graph.EventFilterGraph()
    input_buffer_node = samna.BasicSourceNode_speck2f_event_input_event()
    sink_node = samna.BasicSinkNode_speck2f_event_output_event()
    samna_graph.sequential([input_buffer_node, devkit.get_model_sink_node()])
    samna_graph.sequential([devkit.get_model_source_node(), sink_node])
    samna_graph.sequential([power_source_node, power_buffer_node])
    samna_graph.start()
    stop_watch.set_enable_value(True)

    devkit_cfg = dynapcnn_net.make_config(device=DEVKIT_NAME)
    print(f"Chip layer ordering: {dynapcnn_net.chip_layers_ordering}")
    devkit_cfg.dvs_layer.pass_sensor_events = False
    for i in range(len(dynapcnn_net.chip_layers_ordering)):
        devkit_cfg.cnn_layers[dynapcnn_net.chip_layers_ordering[i]].return_to_zero = True
        devkit_cfg.cnn_layers[dynapcnn_net.chip_layers_ordering[i]].monitor_enable = True
    devkit.get_model().apply_configuration(devkit_cfg)


def measure_chip_power(power_events, loop_duration_s: float, sample_rate_hz: float):
    """(power_w, n_events_ok) from the devkit's PowerMonitor telemetry for one
    run_speck_model() call. power_w = mean of each channel's own mean power (Watts,
    confirmed directly against samna's power-monitoring docs) -- NOT the events' own
    timestamps integrated (a real, ~1000x bug in an earlier version of this file: the power
    monitor reliably delivers sample_rate_hz x n_channels events per second of real loop
    time, i.e. it samples the WHOLE loop, but those events' own .timestamp fields were seen
    to span only 0.02-0.12% of that same loop in every real measurement -- stamped in
    bursts, not evenly, so integrating against them silently undercounts total energy by
    orders of magnitude). n_events_ok is a one-line sanity check: does the number of
    samples collected roughly match what sample_rate_hz x loop_duration_s x n_channels
    predicts? If not, something about this call's power measurement is unreliable and the
    result should not be trusted.
    """
    if not power_events:
        return 0.0, False
    by_channel = defaultdict(list)
    for ev in power_events:
        by_channel[ev.channel].append(float(ev.value))
    power_w = float(sum(np.mean(v) for v in by_channel.values()))
    n_expected = sample_rate_hz * loop_duration_s * len(by_channel)
    n_ok = abs(len(power_events) - n_expected) <= 0.2 * n_expected
    return power_w, n_ok


def run_speck_model(input_tensor, time_steps):
    """input_tensor: (T, C, H, W). Returns (predictions (T, 2) SCALED space, latency_s,
    power_w). latency_s and power_w span the SAME window -- the timestep loop below -- NOT
    the v_mem-reset settling time before it (resetting isn't decoding)."""
    wait_time = args.speck_wait_time
    cores = dynapcnn_net.chip_layers_ordering

    predictions = []
    ema_stages = [torch.zeros(1, N_BINS * 2) for _ in range(TEMPORAL_DECAY_STAGES)]

    sink_node.clear_events()
    for layer in cores:
        set_all_v_mem_to_zeros(chip_factory.get_config_builder(), devkit, layer)
    time.sleep(1)
    power_buffer_node.get_events()  # discard anything that accumulated during the reset settle

    power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
    t0 = time.perf_counter()
    for t in tqdm(range(time_steps), desc="Speck hardware"):
        timeframe = input_tensor[t].unsqueeze(0)
        if timeframe.any():
            event_stream = chip_factory.raster_to_events(timeframe, cores[0], dt=args.speck_raster_dt)
            stop_watch.start(reset=True)
            input_buffer_node.write(event_stream)
        time.sleep(wait_time)

        raw_events = sink_node.get_events()
        output_spikes = [ev for ev in raw_events
                         if isinstance(ev, samna.speck2f.event.Spike) and ev.layer == cores[-1]]
        spikes = torch.zeros(1, N_BINS * 2)
        for feature in range(N_BINS * 2):
            spikes[0, feature] = sum(1 for spk in output_spikes if spk.feature == feature)

        ema = ema_cascade_update(ema_stages, spikes, TEMPORAL_DECAY)
        predictions.append(decode_from_ema(ema).squeeze(0))
    latency_s = time.perf_counter() - t0
    power_monitor.stop_auto_power_measurement()
    power_events = power_buffer_node.get_events()
    power_w, n_events_ok = measure_chip_power(power_events, latency_s, POWER_SAMPLE_RATE_HZ)
    if not n_events_ok:
        print(f"  WARNING: power sample count ({len(power_events)}) doesn't match what "
              f"{POWER_SAMPLE_RATE_HZ} Hz x {latency_s:.2f}s predicts -- this call's power "
              f"reading may be unreliable.")
    print(f"[speck] {time_steps} timestep(s), latency {latency_s * 1000 / time_steps:.4f} ms/step, "
          f"power {power_w * 1e3:.4f} mW")
    return torch.stack(predictions), latency_s, power_w


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


if "speck" in active_impls:
    _, _input_trial, _target_trial = test_dataset[0]
    _input_trial = _input_trial.float()
    _input_frames = _input_trial.unsqueeze(-1).unsqueeze(-1)
    _T = _input_trial.shape[0]
    preds, latency_s, power_w = run_speck_model(_input_frames, _T)
    latency_s_total["speck"] = latency_s
    energy_j_total["speck"] = power_w * latency_s
    energy_method["speck"] = "chip_power_monitor"
    target_phys = unscale_velocity(_target_trial.float(), V_LO, V_HI, V_MARGIN)
    pred_phys_accum["speck"].append(unscale_velocity(preds, V_LO, V_HI, V_MARGIN).numpy())
    if not target_phys_accum:  # only append once -- torch (if also run) already added its own copy
        target_phys_accum.append(target_phys.numpy())

# ---------------------------------------------------------------------------
# 6. Losses: RMSE (both conventions) and CC, per implementation, over every evaluated
#    timestep pooled together.
# ---------------------------------------------------------------------------
target_flat = np.concatenate(target_phys_accum, axis=0)
results = {}
print(f"\n{'=' * 72}\nRESULTS ({len(target_flat)} timestep(s))\n{'=' * 72}")
print(f"  {'Impl':<8} {'RMSE_x':>8} {'RMSE_y':>8} {'pooled':>8} {'mean-ax':>8} "
      f"{'CC_x':>7} {'CC_y':>7} {'CC_avg':>7} {'ms/step':>9} {'mW':>9}")
for impl in active_impls:
    pred_flat = np.concatenate(pred_phys_accum[impl], axis=0)
    err = pred_flat - target_flat
    rmse_x = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    rmse_pooled = float(np.sqrt(np.mean(err ** 2)))
    rmse_mean_of_axes = (rmse_x + rmse_y) / 2  # Oscar/sklearn 'uniform_average' convention
    cc_x = float(np.corrcoef(pred_flat[:, 0], target_flat[:, 0])[0, 1])
    cc_y = float(np.corrcoef(pred_flat[:, 1], target_flat[:, 1])[0, 1])
    cc_avg = (cc_x + cc_y) / 2
    n_steps = pred_flat.shape[0]
    # See latency_n_steps' own definition above for why this can differ from n_steps (torch,
    # after the per-sample timing redesign) or match it exactly (every other impl, unchanged).
    _lat_n = latency_n_steps[impl] if latency_n_steps.get(impl) is not None else n_steps
    ms_per_step = latency_s_total[impl] / _lat_n * 1000
    # energy_j_total[impl] is None only for torch, only after a RAPL wraparound this session
    # (see the "torch" block above) -- speck's energy_j_total is always a real float, it never
    # goes through EnergyMeter/RAPL at all. mw/power_is_real_measurement follow that same None.
    mw = None if energy_j_total[impl] is None else energy_j_total[impl] / latency_s_total[impl] * 1e3
    mw_str = f"{mw:9.4f}" if mw is not None else f"{'n/a':>9}"
    print(f"  {impl:<8} {rmse_x:8.3f} {rmse_y:8.3f} {rmse_pooled:8.3f} {rmse_mean_of_axes:8.3f} "
          f"{cc_x:7.3f} {cc_y:7.3f} {cc_avg:7.3f} {ms_per_step:9.4f} {mw_str}")
    results[impl] = {
        "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_pooled": rmse_pooled,
        "rmse_mean_of_axes": rmse_mean_of_axes,
        "cc_x": cc_x, "cc_y": cc_y, "cc_avg": cc_avg,
        "latency_ms_per_step": ms_per_step,
        "power_mw": mw,
        "energy_j_total": energy_j_total[impl],
        "energy_method": energy_method[impl],
        # False only for torch with energy_method=='proxy_psutil' (RAPL unavailable) -- always
        # True for speck (PowerMonitor is a real measurement by construction) and for torch
        # whenever RAPL was available, wraparound or not: a wraparound makes THIS session's
        # number missing (power_mw is None), not fake, so it's still "real measurement" as a
        # method, just absent as a value for this particular session.
        "power_is_real_measurement": energy_method[impl] != "proxy_psutil",
        # torch only: mean whole-SYSTEM CPU utilization during the measurement window (see the
        # plausibility check above) -- None for speck, and for torch on any run where nothing
        # was sampled (only happens if active_impls somehow reaches here without the torch
        # block having run, which the outer loop structure already prevents).
        "system_cpu_percent_during_measurement": (
            sum(_sys_cpu_pct_samples) / len(_sys_cpu_pct_samples)
            if impl == "torch" and _sys_cpu_pct_samples else None),
        "n_steps": n_steps,
    }

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "session_id": SESSION_ID,
        "experiment": args.experiment,
        "subject": args.subject,
        "checkpoint_path": CHECKPOINT_PATH,
        "dataset_path": DATASET_PATH,
        "active_impls": active_impls,
        "dynapcnn_param_count": DYNAPCNN_PARAM_COUNT,
        "speck_wait_time_s": float(args.speck_wait_time) if "speck" in active_impls else None,
        "results": results,
    }
    out_path = os.path.join(args.output_dir, "speck_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
