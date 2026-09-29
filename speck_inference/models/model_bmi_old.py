"""
Feedforward SNN for hand-velocity decoding.
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

from .init_utils import apply_weight_norm, initialize_snn_model


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

    def build(
        self,
        spike_threshold: float = 1.0,
        override_reset_type: Optional[str] = None,
    ) -> nn.Module:
        """Builds one spiking neuron layer with the requested threshold.
        """
        rst = override_reset_type or self.reset_type

        if self.use_sj:
            v_reset = 0.0 if rst == "hard" else None
            if self.neuron_type == NeuronType.LIF:
                return neuron.LIFNode(v_threshold=spike_threshold, decay_input=True,
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
                    surrogate_grad_fn=grad_fn, tau_mem=self.tau_mem, norm_input=True,
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
                surrogate_grad_fn=grad_fn, tau_mem=self.tau_mem, norm_input=True,
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
        # train_bmi.py's unscale_velocity(): v = lo + (hi-lo)*(scaled-
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

    def forward(self, x_total: torch.Tensor):
        """
        Run the network over T timesteps for a batch of N sequences.
        Membrane state is reset to zero at the start of every call --
        every trial is treated as an independent example (matching how
        the ANN decoders train). An earlier revision supported carrying
        state across calls for a "continuous, never-reset stream"
        training mode; that mode was tried, found substantially worse in
        real comparative results, and removed, along with the
        reset_state parameter and detach_states() that only existed to
        support it -- see module docstring.

        Parameters
        ----------
        x_total : torch.Tensor
            Shape (T, N, num_input_channels) -- time-major input.

        Returns
        -------
        predictions : torch.Tensor, shape (T, N, 2)
            Decoded (x, y) velocity prediction at every timestep.
        final_layer_spikes : torch.Tensor, scalar
            Total spike count in the final layer, summed over the whole call.
        total_spikes : torch.Tensor, scalar
            Total spike count across every layer, summed over the whole call.
        """
        T, N = x_total.shape[0], x_total.shape[1]

        if not self.use_spikingjelly:
            sinabs.utils.reset_states(self)
        else:
            functional.reset_net(self)

        predictions_over_time = []
        final_layer_spikes = torch.zeros(1, device=x_total.device)
        total_spikes = torch.zeros(1, device=x_total.device)

        decay = torch.sigmoid(self._temporal_decay_raw)
        temporal_ema_stages = [torch.zeros((N, 2 * self.n_bins), device=x_total.device)
                                for _ in range(self.temporal_decay_stages)]

        for ts in range(T):
            x = x_total[ts]  # (N, num_input_channels)
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, (_SinabsNeuronLayer, _SJNeuronLayer)):
                    # Spike counts are structurally non-negative; clamp
                    # defensively before accumulating. Under severe
                    # numerical instability (e.g. an unbounded membrane
                    # potential combined with an aggressive learning rate)
                    # this has been observed to go negative, which would
                    # otherwise silently corrupt the spike-sparsity loss
                    # term downstream (see train_bmi.py's
                    # spike_sparsity_lambda) into something that no longer
                    # means "average spikes per neuron per timestep" at
                    # all. --min-vmem should still be set (as it already
                    # is in normal usage) to prevent the instability
                    # itself; this clamp is a backstop, not a substitute.
                    total_spikes = total_spikes + x.sum().clamp(min=0)

            final_layer_spikes = final_layer_spikes + x.sum().clamp(min=0)
            # Cascade: stage 0 fed by this timestep's raw output, each
            # subsequent stage fed by the PREVIOUS stage's own EMA output
            # -- with temporal_decay_stages=1 this is exactly one update,
            # identical to the original single-stage mechanism.
            stage_input = x
            for i in range(self.temporal_decay_stages):
                temporal_ema_stages[i] = decay * temporal_ema_stages[i] + stage_input
                stage_input = temporal_ema_stages[i]
            pred = self.decode_output(temporal_ema_stages[-1])
            predictions_over_time.append(pred)

        predictions = torch.stack(predictions_over_time, dim=0)  # (T, N, 2)
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
    velocity_lo: float = -280.56,
    velocity_hi: float = 316.54,
    velocity_margin: float = 0.05,
    **kwargs,
) -> SNN_Speck:
    """Factory function to create an SNN_Speck model.

    velocity_lo/hi/margin MUST match whatever the dataloader's forward
    scaling and train_bmi.py's unscale_velocity() actually use -- see
    SNN_Speck.__init__()'s neutral_scaled_pred, computed from these three
    values as the exact inverse of unscale_velocity() at physical v=0.

    hidden_dims: list of hidden layer widths (default None -> [512, 256,
    128], reproducing the original fixed 4-layer architecture exactly).
    len(hidden_dims)+1 is the number of layers, and spike_thresholds (if
    given explicitly) must have exactly that many values -- see
    SNN_Speck.__init__()'s own ValueError for the precise, per-call
    requirement.

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

    Extracted here, as a single shared function, after this exact
    exclusion logic needed updating twice already in two separate,
    independently-maintained call sites (test_all_decoders.py's and
    snn_inference_utils.py's own load_snn_model()) -- a third,
    independent copy (for train_bmi.py's --init-weights-from) would mean
    a fourth future change needs to be made in three places instead of
    one. Both of those callers should be migrated to call this function
    too, rather than keep their own inline copies.

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
