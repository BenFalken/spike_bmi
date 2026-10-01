"""
HKM Model -- forked from model_bmi.py specifically for HKM's own,
different training/eval shape: variable-length whole reaching trials
(no fixed 256ms window, no group_size chunking) rather than BMI's fixed,
batchable windows -- see the project-wide discussion of why HKM's short,
independent NWB trials don't fit the 256ms-window/group_size approach
cleanly. Forked rather than shared specifically so model_bmi.py's own,
already-settled, already-deployed BMI behavior stays completely
untouched by anything below -- these two files WILL drift, on purpose.
Everything below is otherwise identical to model_bmi.py -- same
architecture, same fixes, same norm_input/tau_syn/count_ops support --
except forward()'s own reset_state argument, the one deliberate
difference (see below).

Architecture: linear+spiking blocks (default 4: 512 -> 256 -> 128 hidden,
see hidden_dims for customizing this), fed one 4ms timestep at a time
over however many timesteps a given call receives (no longer fixed at
65/256ms here, given HKM's own variable-length trials). Output is
decoded via a causal EMA-smoothed population-vector readout (see
decode_output()) into a continuous (x, y) velocity prediction per
timestep.

Confirmed fixes present in this file (relative to an earlier, pre-fix
version -- kept documented here since neither is obvious from reading
the code alone):
  - spike_threshold now reaches EVERY branch (sinabs LIF, spikingjelly
    LIF/IF, sinabs IAF) -- an earlier revision meant only sinabs IAF
    actually received it; sinabs LIF and both spikingjelly neuron types
    silently fell back to their library's own hardcoded default (1.0),
    regardless of what spike_thresholds/--spike-thresholds specified.
    Confirmed directly: constructed two LIF neurons with
    spike_threshold=0.5 and 5.0 and found IDENTICAL vmem trajectories
    and spike counts before this fix -- the argument had no effect
    whatsoever, not just a subtle one. This project's own production
    checkpoints all use neuron_type='iaf', which was never affected.
  - `positions` changed from geometric decay (0.5**arange(n_bins), which
    left bins 4+ contributing almost nothing -- effectively only ~4 of
    18 bins were doing any work) to torch.linspace(0, 1, n_bins) --
    genuine, uniformly-spaced population-vector semantics, full n_bins
    resolution actually usable.

Cross-batch state, THE ONE DELIBERATE DIFFERENCE FROM model_bmi.py: every
forward() call still resets membrane state at the START by default
(reset_state=True), same as model_bmi.py -- but reset_state is now an
explicit, optional argument again, so a caller CAN suppress that reset
for a specific call. This is NOT a reintroduction of the old "continuous,
never-reset stream" TRAINING mode that was tried, found worse, and
removed from model_bmi.py -- training here still resets per-trial by
default, same as BMI. This exists for HKM's own, separate, INFERENCE-time
question: whether test evaluation should reset state at each trial
boundary (reset_state=True per call, one call per trial -- the default,
and the direct HKM analogue of how BMI/ANN/KF/WF are all evaluated) or
run the whole stitched-together test stream through as one continuous
decode with no resets at all (reset_state=False, whole stream in one
call) -- a genuinely different question from the training-mode one
model_bmi.py's own docstring describes, worth being able to test
directly rather than assume the answer. See forward()'s own docstring
for the concrete usage patterns.

norm_input (LIF only -- IAF has no leak/normalization concept) controls
whether incoming input is divided by tau_mem before accumulating
(sinabs' own default: True) -- see NeuronFactory.build()'s own docstring
for the real, confirmed depth-attenuation consequence this has.

tau_syn (BOTH IAF and LIF -- confirmed directly via inspect.signature,
not assumed: sinabs' IAF constructor accepts tau_syn too, not just LIF)
adds a SECOND filtering stage before the membrane potential itself:
input -> synaptic current (decaying at tau_syn) -> membrane potential
(decaying at tau_mem, for LIF; accumulating without decay for IAF).
Default None matches sinabs' own default exactly.

count_ops (forward()'s own optional argument, default False, zero
overhead when off): accumulates raw effective-operation counts (MAC,
ACC, elementwise) inline during the same forward pass used for training/
eval, rather than via a separate external estimator re-walking the model
afterward -- see forward()'s own docstring for why counting inline here
specifically was necessary (this function calls itself once per
timestep internally, and the EMA cascade + decode_output() both
contribute real additional operations an external re-walk couldn't see
without re-deriving this function's own logic a second time). Feeds
op_energy_estimate.py's finalize_snn_ops(), the same MAC/ACC framework
already used for every other decoder in this project. MUST stay off
during training -- only test_all_decoders.py's evaluation call should
ever turn it on.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union
from enum import Enum

import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl
import sinabs.utils
import sinabs.activation
from sinabs.activation import (
    Gaussian,
    Heaviside,
    MultiGaussian,
    PeriodicExponential,
    SingleExponential,
)
from spikingjelly.activation_based import functional, neuron, surrogate

from .init_utils import initialize_snn_model


# ---------------------------------------------------------------------------
# Optional Exodus acceleration
# ---------------------------------------------------------------------------
try:
    import sinabs.exodus.layers as exodus_layers

    EXODUS_AVAILABLE = True
    print("Sinabs Exodus detected -- using CUDA-accelerated layers")
    print("  Note: Exodus requires soft reset (MembraneSubtract) for all neurons")
except ImportError:
    EXODUS_AVAILABLE = False


class NeuronType(Enum):
    IAF = "iaf"
    LIF = "lif"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def get_reset_fn(reset_type: str, force_soft: bool = False):
    """Return a sinabs reset function. reset_type: 'hard' or 'soft'.
    force_soft: always return soft reset (required by Exodus)."""
    if force_soft or reset_type == "soft":
        return sinabs.activation.MembraneSubtract()
    if reset_type == "hard":
        return sinabs.activation.MembraneReset()
    raise ValueError(f"Unknown reset_type {reset_type!r}. Choose 'hard' or 'soft'.")


def get_surrogate_grad_fn(surrogate_type: Union[str, dict, Callable]) -> Callable:
    """Return an instantiated sinabs surrogate-gradient function."""
    if callable(surrogate_type):
        return surrogate_type

    if isinstance(surrogate_type, dict):
        fn_type = surrogate_type.get("type", "periodic_exponential")
        params = {k: v for k, v in surrogate_type.items() if k != "type"}
    else:
        fn_type = surrogate_type
        params = {}

    _map = {
        "periodic_exponential": PeriodicExponential,
        "single_exponential": SingleExponential,
        "gaussian": Gaussian,
        "multi_gaussian": MultiGaussian,
        "heaviside": Heaviside,
    }
    if fn_type not in _map:
        raise ValueError(f"Unknown surrogate gradient {fn_type!r}. Available: {list(_map)}")
    return _map[fn_type](**params)


# ---------------------------------------------------------------------------
# Neuron-layer factory
# ---------------------------------------------------------------------------

class NeuronFactory:
    """Builds a single spiking neuron layer with the requested backend/type."""

    def __init__(
        self,
        neuron_type: NeuronType,
        use_spikingjelly: bool,
        use_exodus: bool,
        reset_type: str,
        surrogate_grad: Union[str, dict, Callable],
        tau_mem: float,
        min_vmem: float,
        spike_fn: Optional[Callable],
        use_squeeze: bool,
        tau_syn: Optional[float] = None,
        norm_input: bool = True,
    ):
        self.use_squeeze = use_squeeze
        self.neuron_type = neuron_type
        self.use_sj = use_spikingjelly
        self.use_exodus = use_exodus and EXODUS_AVAILABLE
        self.reset_type = reset_type
        self.surrogate_grad = surrogate_grad
        self.tau_mem = tau_mem
        self.min_vmem = min_vmem
        self.spike_fn = spike_fn
        self.tau_syn = tau_syn
        self.norm_input = norm_input

    def build(
        self,
        spike_threshold: float = 1.0,
        override_reset_type: Optional[str] = None,
    ) -> nn.Module:
        """Builds one spiking neuron layer with the requested threshold.

        norm_input (LIF only -- IAF has no leak/normalization concept)
        controls whether incoming input is divided by tau_mem before
        accumulating (sinabs' own default: True). This has a real,
        confirmed, non-obvious consequence for deep networks: with
        norm_input=True, EVERY layer divides whatever it receives by
        tau_mem again, compounding multiplicatively with depth -- at
        tau_mem=100 on a real 5-layer network, this measurably produced
        complete signal death by layer 2-3 regardless of how low earlier
        layers' thresholds were set, since the problem isn't any single
        layer's threshold but the exponential-with-depth attenuation
        itself. norm_input=False avoids the division (confirmed: a brief
        input pulse's membrane trace persisted for 20+ timesteps at
        tau_mem=100 with norm_input=False, vs. decaying to zero in ~10
        steps at tau_mem=1) -- genuine long-timescale memory without the
        input being divided away. Whether that alone resolves the depth-
        attenuation problem for a REAL multi-layer network is NOT yet
        confirmed here (isolated 2-neuron tests gave inconsistent, still-
        being-investigated results depending on exact calibration) --
        this parameter exists so that question can be tested directly
        against the real architecture and real data, via
        diagnose_layer_thresholds.py --norm-input, rather than guessed at.
        """
        rst = override_reset_type or self.reset_type

        if self.use_sj:
            v_reset = 0.0 if rst == "hard" else None
            if self.neuron_type == NeuronType.LIF:
                return neuron.LIFNode(v_threshold=spike_threshold, decay_input=self.norm_input,
                                       step_mode="s", v_reset=v_reset, surrogate_function=surrogate.ATan())
            return neuron.IFNode(v_threshold=spike_threshold, step_mode="s", v_reset=v_reset,
                                  surrogate_function=surrogate.ATan())

        force_soft = self.use_exodus
        reset_fn = get_reset_fn(rst, force_soft=force_soft)
        grad_fn = get_surrogate_grad_fn(self.surrogate_grad)

        if self.use_squeeze:
            if self.neuron_type == NeuronType.LIF:
                cls = exodus_layers.LIFSqueeze if self.use_exodus else sl.LIFSqueeze
                return cls(
                    spike_threshold=spike_threshold, min_v_mem=self.min_vmem,
                    surrogate_grad_fn=grad_fn, tau_mem=self.tau_mem, norm_input=self.norm_input,
                    tau_syn=self.tau_syn,
                    reset_fn=reset_fn, spike_fn=self.spike_fn, num_timesteps=1,
                )
            cls = exodus_layers.IAFSqueeze if self.use_exodus else sl.IAFSqueeze
            return cls(
                spike_threshold=spike_threshold, min_v_mem=self.min_vmem, surrogate_grad_fn=grad_fn,
                tau_syn=self.tau_syn,
                reset_fn=reset_fn, spike_fn=self.spike_fn, num_timesteps=1,
            )

        if self.neuron_type == NeuronType.LIF:
            cls = exodus_layers.LIF if self.use_exodus else sl.LIF
            return cls(
                spike_threshold=spike_threshold, min_v_mem=self.min_vmem,
                surrogate_grad_fn=grad_fn, tau_mem=self.tau_mem, norm_input=self.norm_input,
                tau_syn=self.tau_syn,
                reset_fn=reset_fn, spike_fn=self.spike_fn,
            )
        cls = exodus_layers.IAF if self.use_exodus else sl.IAF
        return cls(
            spike_threshold=spike_threshold, min_v_mem=self.min_vmem, surrogate_grad_fn=grad_fn,
            tau_syn=self.tau_syn,
            reset_fn=reset_fn, spike_fn=self.spike_fn,
        )


# ---------------------------------------------------------------------------
# Thin wrappers that own a neuron layer, exposing .vmem()/.reset_states()
# ---------------------------------------------------------------------------

class _SinabsNeuronLayer(nn.Module):
    """Wraps a sinabs neuron layer, handling the unsqueeze(1)/squeeze(1)
    needed for 1-timestep-at-a-time processing on linear layers (standard,
    non-Squeeze case), or passing through directly for Squeeze layers."""

    def __init__(self, neuron_layer: nn.Module, last_layer_reset: bool = False, is_squeeze: bool = False):
        super().__init__()
        self.neuron = neuron_layer
        self.last_layer_reset = last_layer_reset
        self.is_squeeze = is_squeeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_layer_reset:
            self.neuron.reset_states()
        if self.is_squeeze:
            return self.neuron(x)
        return self.neuron(x.unsqueeze(1)).squeeze(1)

    def vmem(self) -> torch.Tensor:
        return self.neuron.v_mem.clone()

    def reset_states(self):
        self.neuron.reset_states()


class _SJNeuronLayer(nn.Module):
    """Wraps a SpikingJelly IF/LIF node, handling last-layer reset."""

    def __init__(self, neuron_layer: nn.Module, last_layer_reset: bool = False):
        super().__init__()
        self.neuron = neuron_layer
        self.last_layer_reset = last_layer_reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_layer_reset:
            self.neuron.v = self.neuron.v * 0.0
        return self.neuron(x)

    def vmem(self) -> torch.Tensor:
        return self.neuron.v.clone()

    def reset_states(self):
        self.neuron.v = self.neuron.v * 0.0


# ---------------------------------------------------------------------------
# Block builder -- linear only, this architecture never uses conv/pool
# ---------------------------------------------------------------------------

def add_linear_block(
    layers: List[nn.Module],
    in_features: int,
    out_features: int,
    factory: NeuronFactory,
    spike_threshold: float = 1.0,
    override_reset_type: Optional[str] = None,
    last_layer_reset: bool = False,
) -> List[nn.Module]:
    """Append: Linear -> NeuronLayer."""
    layers.append(nn.Linear(in_features, out_features, bias=False))
    raw = factory.build(spike_threshold=spike_threshold, override_reset_type=override_reset_type)
    if factory.use_sj:
        layers.append(_SJNeuronLayer(raw, last_layer_reset=last_layer_reset))
    else:
        layers.append(_SinabsNeuronLayer(raw, last_layer_reset=last_layer_reset, is_squeeze=factory.use_squeeze))
    return layers


def reset_vmems(network: nn.Module):
    """Reset all spiking neuron states in *network*."""
    for m in network.modules():
        if isinstance(m, (_SinabsNeuronLayer, _SJNeuronLayer)):
            m.reset_states()


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SNN_Speck(nn.Module):
    """
    Feedforward spiking network for BMI hand-velocity decoding.
    """

    def __init__(
        self,
        use_spikingjelly: bool,
        last_layer_reset: bool,
        min_vmem: float,
        spike_fn: Optional[Callable],
        neuron_type: NeuronType = NeuronType.LIF,
        tau_mem: float = 1.0,
        reset_type: str = "hard",
        final_layer_reset_type: Optional[str] = None,
        surrogate_grad: Union[str, dict, Callable] = "periodic_exponential",
        use_exodus: bool = False,
        use_iaf_squeeze: bool = False,
        n_bins: int = 18,
        spike_thresholds: Optional[List[float]] = None,
        temporal_decay_init: float = 0.8,
        learnable_temporal_decay: bool = True,
        temporal_decay_stages: int = 1,
        num_input_channels: int = 96,
        hidden_dims: Optional[List[int]] = None,
        tau_syn: Optional[float] = None,
        norm_input: bool = True,
        velocity_lo: float = -280.56,
        velocity_hi: float = 316.54,
        velocity_margin: float = 0.05,
    ):
        super().__init__()
        self.use_spikingjelly = use_spikingjelly
        self.use_iaf_squeeze = use_iaf_squeeze
        self.n_bins = n_bins
        out_rst = final_layer_reset_type or reset_type

        # Hidden dimensions can be customized as needed, but the original structure remains as default.
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [512, 256, 128]
        layer_widths = [num_input_channels] + hidden_dims + [2 * n_bins]
        n_layers = len(layer_widths) - 1

        # Spike threshold pattern can also be modified, leaving empty triggers default.
        thresholds = spike_thresholds or ([2.0] + [1.0] * (n_layers - 1))
        if len(thresholds) != n_layers:
            raise ValueError(
                f"spike_thresholds must have exactly {n_layers} values (one per layer, "
                f"given hidden_dims={hidden_dims} -> {n_layers} layers total), "
                f"got {len(thresholds)}")

        factory = NeuronFactory(
            neuron_type=neuron_type,
            use_spikingjelly=use_spikingjelly,
            use_exodus=use_exodus,
            reset_type=reset_type,
            surrogate_grad=surrogate_grad,
            tau_mem=tau_mem,
            min_vmem=min_vmem,
            spike_fn=spike_fn,
            use_squeeze=use_iaf_squeeze,
            tau_syn=tau_syn,
            norm_input=norm_input,
        )

        # Dynamically-built SNN
        raw_layers: List[nn.Module] = []
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            add_linear_block(
                raw_layers, layer_widths[i], layer_widths[i + 1], factory,
                spike_threshold=thresholds[i],
                override_reset_type=out_rst if is_last else None,
                last_layer_reset=last_layer_reset if is_last else False,
            )
        self.layers = nn.ModuleList(raw_layers)
        self.total_neuron_units = sum(layer_widths[1:])

        positions_init = torch.linspace(0.0, 1.0, n_bins, dtype=torch.float32)
        self.register_buffer("positions", positions_init)

        # EMA readout: bounded, learnable scalar
        init_logit = torch.logit(torch.tensor(float(temporal_decay_init)))
        self._temporal_decay_raw = nn.Parameter(init_logit, requires_grad=learnable_temporal_decay)

        # Number of CASCADED EMA stages feeding the readout -- default 1
        # reproduces the original single-stage EMA exactly (confirmed
        # directly: with 1 stage, the cascade loop in forward() reduces
        # to precisely the old temporal_ema = decay*temporal_ema + x
        # update, byte-for-byte). >1 stages chains multiple EMAs (all
        # sharing this SAME decay parameter), which is a well-established
        # signal-processing technique for building a gamma/Erlang-shaped
        # kernel -- verified numerically before implementing here: unlike
        # a single EMA (peaked at lag 0, monotonically decreasing), a
        # 2-stage cascade with decay=0.9 peaks at lag 8 with a 9-step
        # plateau before decaying, and more stages push the peak further
        # out and widen the plateau further. This lets the network
        # weight a RANGE of recent windows near-equally before tapering,
        # rather than always weighting the single most recent timestep
        # most heavily.
        if temporal_decay_stages < 1:
            raise ValueError(f"temporal_decay_stages must be >= 1, got {temporal_decay_stages}")
        self.temporal_decay_stages = temporal_decay_stages

        # The SCALED-space value that unscales to a PHYSICAL velocity of
        # exactly 0.0 -- used as decode_output()'s fallback for whichever
        # individual samples in a batch produce zero spikes in a given
        # axis at a given timestep (no accumulated population-vector
        # signal to decode). Computed as the exact algebraic inverse of
        # train_snn.py's unscale_velocity(): v = lo + (hi-lo)*(scaled-
        # margin)/(1-2*margin), solved for scaled at v=0. MUST be built
        # from the SAME velocity_lo/hi/margin the dataloader's forward
        # scaling and unscale_velocity() use, or "no spikes" stops
        # actually meaning "predict no movement" and silently means some
        # other, non-zero physical velocity instead.
        #
        # This replaces an earlier fallback (an unconditional
        # denominator clamp) that put every no-spike sample at SCALED
        # 0.0, not the value corresponding to physical 0.0 -- with this
        # project's actual velocity_lo=-280.56, that unscaled to
        # roughly -313.7, a value MORE EXTREME than anything the
        # network was ever calibrated to predict (v_lo itself), for
        # what should have been the single most conservative, least
        # committal output the network could produce.
        neutral_scaled = velocity_margin - velocity_lo * (1 - 2 * velocity_margin) / (velocity_hi - velocity_lo)
        self.register_buffer("neutral_scaled_pred",
                              torch.tensor(float(neutral_scaled), dtype=torch.float32))

    @property
    def temporal_decay(self) -> float:
        """Current EMA decay in (0, 1)."""
        return torch.sigmoid(self._temporal_decay_raw).item()

    def decode_output(self, acc: torch.Tensor) -> torch.Tensor:
        """Turn one timestep's EMA-smoothed final-layer output into a
        decoded (x, y) prediction. acc: (N, 2*n_bins).

        Falls back to neutral_scaled_pred (see __init__) -- NOT the
        previous behavior of an unconditional denominator clamp landing
        at scaled 0.0 -- for whichever individual samples in the batch
        have zero accumulated activity in a given axis. Checked and
        applied PER SAMPLE (via torch.where, not a single scalar
        branch), so a batch where some sequences have spiked and others
        haven't yet (the normal case at the start of a trial) is handled
        correctly for each one independently.
        """
        n = self.n_bins
        x_acc = acc[:, :n]
        y_acc = acc[:, n:2 * n]
        x_raw_total = x_acc.sum(dim=1)
        y_raw_total = y_acc.sum(dim=1)
        x_has_signal = x_raw_total > 0
        y_has_signal = y_raw_total > 0
        x_total = x_raw_total.clamp(min=1)  # backstop against div-by-zero only; the
        y_total = y_raw_total.clamp(min=1)  # actual no-signal case is handled by torch.where below
        pos = self.positions.to(acc.device)
        pred_x_raw = (x_acc * pos).sum(dim=1) / x_total
        pred_y_raw = (y_acc * pos).sum(dim=1) / y_total
        neutral = self.neutral_scaled_pred.to(acc.device)
        pred_x = torch.where(x_has_signal, pred_x_raw, neutral)
        pred_y = torch.where(y_has_signal, pred_y_raw, neutral)
        return torch.stack([pred_x, pred_y], dim=-1)

    def forward(self, x_total: torch.Tensor, count_ops: bool = False, reset_state: bool = True):
        """
        Run the network over T timesteps for a batch of N sequences.

        reset_state (default True, matching model_bmi.py's own
        unconditional behavior exactly): whether to reset membrane state
        at the START of this call. True is correct for training (one
        trial, one independent example -- unchanged from model_bmi.py)
        and for the default, "reset per trial" evaluation mode. Pass
        False to carry state in FROM WHATEVER IT WAS at the end of the
        previous call -- e.g. for a continuous, no-reset decode across
        multiple stitched-together test trials called one at a time, or
        for a single call already covering a full stitched stream where
        the caller wants no mid-stream reset at all. This is NOT a
        reintroduction of the old "continuous, never-reset stream"
        TRAINING mode model_bmi.py's own docstring describes as tried,
        found worse, and removed -- training here still resets per-trial
        by default, same as BMI; this is a separate, inference-time-only
        question specific to HKM's own stitched-test-stream evaluation
        (see module docstring).

        Parameters
        ----------
        x_total : torch.Tensor
            Shape (T, N, num_input_channels) -- time-major input.
        count_ops : bool, default False
            If True, ALSO accumulates and returns raw effective-operation
            counts (MAC, ACC, elementwise) across every layer, the EMA
            cascade, and decode_output() -- see
            op_energy_estimate.py's finalize_snn_ops() for converting
            these into memory accesses and an energy estimate, the same
            MAC/ACC framework already used for every other decoder in
            this project. Counted HERE, inline, rather than by a
            separate external estimator re-walking the model afterward
            (which is what every OTHER decoder in this project uses) --
            an earlier attempt at an external estimator had to be
            discarded once this function's real structure was seen: it
            calls itself once per timestep internally (not once per
            sample externally), and the EMA cascade + decode_output()
            both contribute real additional operations beyond the
            four-layer IAF stack itself, neither of which an external
            re-walk could see without re-deriving this function's own
            logic a second time, with the drift risk that implies.

            Defaults to False and adds ZERO overhead to the normal
            training/eval path when off (every accumulation below is
            behind an `if count_ops:` guard) -- this MUST stay off
            during training; only test_all_decoders.py's evaluation call
            should ever turn it on.
        reset_state : bool, default True
            Reset membrane state at the start of this call -- see above.

        Returns
        -------
        predictions : torch.Tensor, shape (T, N, 2)
            Decoded (x, y) velocity prediction at every timestep.
        final_layer_spikes : torch.Tensor, scalar
            Total spike count in the final layer, summed over the whole call.
        total_spikes : torch.Tensor, scalar
            Total spike count across every layer, summed over the whole call.
        op_counts : dict, ONLY returned when count_ops=True
            {'mac': int, 'acc': float, 'elementwise': int} -- raw,
            UNCONVERTED operation counts for the whole call (already
            summed over every timestep and every sample in the batch;
            pass n_samples = T * N to finalize_snn_ops() alongside these).
        """
        T, N = x_total.shape[0], x_total.shape[1]

        if reset_state:
            if not self.use_spikingjelly:
                sinabs.utils.reset_states(self)
            else:
                functional.reset_net(self)

        predictions_over_time = []
        final_layer_spikes = torch.zeros(1, device=x_total.device)
        total_spikes = torch.zeros(1, device=x_total.device)

        if count_ops:
            op_mac = 0
            op_acc = 0.0
            op_elementwise = 0

        decay = torch.sigmoid(self._temporal_decay_raw)
        temporal_ema_stages = [torch.zeros((N, 2 * self.n_bins), device=x_total.device)
                                for _ in range(self.temporal_decay_stages)]

        for ts in range(T):
            x = x_total[ts]  # (N, num_input_channels)
            linear_seen = 0
            for layer in self.layers:
                if count_ops and isinstance(layer, nn.Linear):
                    out_features = layer.out_features
                    if linear_seen == 0:
                        # First Linear: raw, continuous-valued input (this
                        # timestep's MUA feature vector, not a spike train
                        # yet) -- MAC, sparsity-aware via nnz, same rule as
                        # every other decoder's own first layer.
                        op_mac += int(torch.count_nonzero(x).item()) * out_features
                    else:
                        # Every later Linear: input is the PRECEDING neuron
                        # layer's actual spike output -- genuinely spike-
                        # driven, so ACC, tallied from the SUM of spike
                        # VALUES (not just nnz) since this model uses
                        # MultiSpike -- a neuron can emit more than one
                        # spike in a single timestep, and each one needs a
                        # separate unit of accumulate work downstream.
                        op_acc += float(x.sum().item()) * out_features
                    linear_seen += 1
                x = layer(x)
                if isinstance(layer, (_SinabsNeuronLayer, _SJNeuronLayer)):
                    # Spike counts are structurally non-negative; clamp
                    # defensively before accumulating. Under severe
                    # numerical instability (e.g. an unbounded membrane
                    # potential combined with an aggressive learning rate)
                    # this has been observed to go negative, which would
                    # otherwise silently corrupt the spike-sparsity loss
                    # term downstream (see train_snn.py's
                    # spike_sparsity_lambda) into something that no longer
                    # means "average spikes per neuron per timestep" at
                    # all. --min-vmem should still be set (as it already
                    # is in normal usage) to prevent the instability
                    # itself; this clamp is a backstop, not a substitute.
                    total_spikes = total_spikes + x.sum().clamp(min=0)
                    if count_ops:
                        units = x.shape[-1]
                        if isinstance(layer, _SinabsNeuronLayer):
                            # Synaptic current decay (i_syn[t] = decay*i_syn[t-1]
                            # + I_in[t]): a REAL multiply against a continuous
                            # decay constant, every neuron, every timestep,
                            # regardless of spike activity -- sinabs-SPECIFIC
                            # (tau_syn), not applicable to the SpikingJelly
                            # backend, which this project's own NeuronFactory
                            # never passes tau_syn to (see NeuronFactory.build()).
                            op_mac += N * units
                        # Membrane update (v_mem += i_syn, IAF has no leak of
                        # its own) + threshold compare + MembraneSubtract reset
                        # -- a small per-neuron add-and-compare, not a
                        # weighted-sum operation, for EITHER backend.
                        op_elementwise += 2 * N * units

            final_layer_spikes = final_layer_spikes + x.sum().clamp(min=0)
            # Cascade: stage 0 fed by this timestep's raw output, each
            # subsequent stage fed by the PREVIOUS stage's own EMA output
            # -- with temporal_decay_stages=1 this is exactly one update,
            # identical to the original single-stage mechanism.
            stage_input = x
            for i in range(self.temporal_decay_stages):
                if count_ops:
                    # decay*prev + input: one multiply+add per element of
                    # this (N, 2*n_bins) stage tensor.
                    op_mac += N * (2 * self.n_bins)
                temporal_ema_stages[i] = decay * temporal_ema_stages[i] + stage_input
                stage_input = temporal_ema_stages[i]
            pred = self.decode_output(temporal_ema_stages[-1])
            if count_ops:
                # decode_output(): a position-weighted dot product over
                # n_bins per axis per sample (n_bins MACs each for x and
                # y), plus the two sum(dim=1) normalization reductions
                # (counted as elementwise adds, not MAC -- no weight
                # multiply involved).
                op_mac += N * 2 * self.n_bins
                op_elementwise += N * 2 * self.n_bins
            predictions_over_time.append(pred)

        predictions = torch.stack(predictions_over_time, dim=0)  # (T, N, 2)
        if count_ops:
            return predictions, final_layer_spikes, total_spikes, {
                'mac': op_mac, 'acc': op_acc, 'elementwise': op_elementwise}
        return predictions, final_layer_spikes, total_spikes


def create_model(
    use_spikingjelly: bool,
    last_layer_reset: bool,
    weight_init: str,
    spike_fn,
    min_vmem,
    neuron_type: str = "lif",
    tau_mem: float = 1.0,
    reset_type: str = "hard",
    final_layer_reset_type=None,
    surrogate_grad: Union[str, dict, Callable] = "periodic_exponential",
    use_exodus: bool = False,
    use_iaf_squeeze: bool = False,
    n_bins: int = 18,
    spike_thresholds: Optional[List[float]] = None,
    temporal_decay_init: float = 0.8,
    learnable_temporal_decay: bool = True,
    temporal_decay_stages: int = 1,
    num_input_channels: int = 96,
    hidden_dims: Optional[List[int]] = None,
    tau_syn: Optional[float] = None,
    norm_input: bool = True,
    velocity_lo: float = -280.56,
    velocity_hi: float = 316.54,
    velocity_margin: float = 0.05,
    **kwargs,
) -> SNN_Speck:
    """Factory function to create an SNN_Speck model.

    velocity_lo/hi/margin MUST match whatever the dataloader's forward
    scaling and train_snn.py's unscale_velocity() actually use -- see
    SNN_Speck.__init__()'s neutral_scaled_pred, computed from these three
    values as the exact inverse of unscale_velocity() at physical v=0.

    hidden_dims: list of hidden layer widths (default None -> [512, 256,
    128], reproducing the original fixed 4-layer architecture exactly).
    len(hidden_dims)+1 is the number of layers, and spike_thresholds (if
    given explicitly) must have exactly that many values -- see
    SNN_Speck.__init__()'s own ValueError for the precise, per-call
    requirement.

    norm_input: LIF only (ignored for IAF, which has no leak to
    normalize against) -- see NeuronFactory.build()'s docstring for the
    real, confirmed consequence this has for deep networks with large
    tau_mem.
    """
    try:
        neuron_enum = NeuronType(neuron_type.lower())
    except ValueError:
        raise ValueError(f"neuron_type must be one of {[t.value for t in NeuronType]}, "
                          f"got {neuron_type!r}")

    model = SNN_Speck(
        use_spikingjelly=use_spikingjelly,
        last_layer_reset=last_layer_reset,
        min_vmem=min_vmem,
        spike_fn=spike_fn,
        neuron_type=neuron_enum,
        tau_mem=tau_mem,
        reset_type=reset_type,
        final_layer_reset_type=final_layer_reset_type,
        surrogate_grad=surrogate_grad,
        use_exodus=use_exodus,
        use_iaf_squeeze=use_iaf_squeeze,
        n_bins=n_bins,
        spike_thresholds=spike_thresholds,
        temporal_decay_init=temporal_decay_init,
        learnable_temporal_decay=learnable_temporal_decay,
        temporal_decay_stages=temporal_decay_stages,
        num_input_channels=num_input_channels,
        hidden_dims=hidden_dims,
        tau_syn=tau_syn,
        norm_input=norm_input,
        velocity_lo=velocity_lo,
        velocity_hi=velocity_hi,
        velocity_margin=velocity_margin,
    )

    initialize_snn_model(model, weight_init)
    return model


def load_model_weights(model: SNN_Speck, state_dict: dict, neuron_type: str,
                        source_description: str = "checkpoint") -> None:
    """Loads ONLY weight/parameter values into `model` from `state_dict`,
    correctly excluding sinabs' lazily-shaped STATE buffers (not learned
    parameters) rather than letting load_state_dict fail on them.

    Shared by every checkpoint loader (inference's load_snn_model() and
    train_snn.py's --init-weights-from).

    v_mem/i_syn are lazily shaped for EVERY neuron type (sinabs only
    gives them their real shape the first time forward() actually runs
    with real data -- a freshly-constructed model sits at a degenerate
    placeholder shape, which load_state_dict refuses to overwrite with a
    checkpoint's differently-shaped, already-initialized values).

    neuron_type is kept as a parameter for call-site compatibility, but
    no longer changes this function's behavior -- it previously
    determined an ALIF-specific extra exclusion (.b/.spike_threshold, ALIF's
    adaptation state and its derived, per-step threshold), removed along
    with ALIF support itself.

    Raises RuntimeError on anything that looks like a genuine
    architecture mismatch (a missing Linear.weight, an unexpected key)
    rather than silently loading a partially-wrong model.
    """
    exclude_suffixes = ['.v_mem', '.neuron.v', '.i_syn']
    filtered_state_dict = {k: v for k, v in state_dict.items()
                            if not any(k.endswith(suffix) for suffix in exclude_suffixes)}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    unexpected_missing = [k for k in missing
                           if not (any(k.endswith(suffix) for suffix in exclude_suffixes)
                                   or k == 'neutral_scaled_pred')]
    if unexpected_missing:
        raise RuntimeError(
            f"{source_description}: load_state_dict is missing key(s) beyond the "
            f"expected {exclude_suffixes} state buffers: {unexpected_missing} -- this "
            f"suggests a real architecture mismatch, not the lazy-state-buffer issue "
            f"this filtering is meant to handle.")
    if unexpected:
        raise RuntimeError(
            f"{source_description}: checkpoint has unexpected key(s) not present in the "
            f"current model_bmi.py architecture: {unexpected}")
