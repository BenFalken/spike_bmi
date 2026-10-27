"""
Recurrent LSNN velocity decoder (train_snn.py --arch lsnn), ported from the
speech decoder (speechBCI SNNDecoder, rsnn_native.py, final model "Model B"):

    input (96 channels, one 4 ms bin per step)
      -> recurrent spiking layers: integrate-and-fire (no leak), soft reset,
         at most --lsnn-max-spikes spikes per step, a per-neuron synaptic
         filter (time constants log-spaced over --lsnn-tau-syn-ms, trained),
         and threshold adaptation in a fixed random --lsnn-adaptive-fraction
         of the neurons (threshold 1 + adapt_scale * b, b decaying over
         --lsnn-tau-adapt-ms)
      -> leaky, non-spiking readout neurons, one per velocity axis, reading
         the last layer's spikes; their membrane potential is the (scaled)
         velocity prediction at every step (--lsnn-readout-all: every layer's
         spikes, not only the last). --lsnn-out-stages S > 1 chains S
         leaky stages per axis (each its own trained time constant, starting at
         --lsnn-out-tau-ms), a smoother output, as the SNN's cascaded EMA.

All time constants are given in milliseconds and converted to 4 ms steps, so
they keep the speech model's physical values (speech steps were 20 ms).

Same interface as models/model_bmi.py's SNN_Speck: forward(x) with x of
shape (T, N, channels) returns (predictions (T, N, 2), final-layer spikes,
total spikes); layer_spikes, layer_widths and total_neuron_units feed
train_snn.py's spike-rate penalty and logging. The state is reset at the
start of every call (every trial is independent), as for the SNN, unless
reset_state=False (inference on hkm), which carries it over from the previous
call. count_ops=True also returns operation counts in the SNN's convention
(inference/op_energy_estimate.py finalize_snn_ops): the first layer's input
as multiply-adds per non-zero input, spike-driven synapses (recurrent,
between layers, readout) as accumulates per spike, and the neuron updates
(synaptic filter, adaptation, readout leak) as multiply-adds.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

STEP_MS = 4.0


class MaxSpike(torch.autograd.Function):
    """Spike count per step: floor(v / threshold), clipped to [0, max_spikes].
    Backward: a single-exponential surrogate around the threshold, scaled by
    1 / threshold (as sinabs' SingleExponential with MultiSpike)."""

    @staticmethod
    def forward(ctx, v, threshold, max_spikes, width):
        ctx.save_for_backward(v, threshold)
        ctx.width = width
        return torch.clamp(torch.floor(v / threshold), 0, max_spikes)

    @staticmethod
    def backward(ctx, grad_out):
        v, threshold = ctx.saved_tensors
        sg = torch.exp(-torch.abs(v - threshold) / ctx.width) / threshold
        return grad_out * sg, None, None, None


class LSNNLayer(nn.Module):
    """One recurrent layer of integrate-and-fire neurons, a fraction adaptive."""

    def __init__(self, n_in: int, n_out: int, tau_syn_steps: List[float], adaptive_fraction: float,
                 tau_adapt_steps: float, adapt_scale: float, max_spikes: int, rec_init_scale: float,
                 max_vmem: float, min_vmem: float, seed: int):
        super().__init__()
        self.width = n_out
        self.ff = nn.Linear(n_in, n_out, bias=False)
        nn.init.kaiming_uniform_(self.ff.weight)
        self.rec = nn.Linear(n_out, n_out, bias=False)
        nn.init.orthogonal_(self.rec.weight)
        with torch.no_grad():
            self.rec.weight.mul_(rec_init_scale)
        lo, hi = tau_syn_steps
        self.log_tau_syn = nn.Parameter(torch.log(torch.logspace(math.log10(lo), math.log10(hi), n_out)))
        g = torch.Generator().manual_seed(seed)
        mask = torch.zeros(n_out)
        mask[torch.randperm(n_out, generator=g)[:int(round(adaptive_fraction * n_out))]] = 1.0
        self.register_buffer("adapt_mask", mask)
        self.alpha_adapt = math.exp(-1.0 / tau_adapt_steps)
        self.adapt_scale, self.max_spikes = adapt_scale, max_spikes
        self.max_vmem, self.min_vmem = max_vmem, min_vmem

    def init_state(self, n: int, device) -> dict:
        z = lambda: torch.zeros(n, self.width, device=device)
        return {"i": z(), "v": z(), "b": z(), "s": z()}

    def step(self, current: torch.Tensor, st: dict) -> torch.Tensor:
        """One 4 ms step given this step's feedforward current; returns spikes."""
        current = current + self.rec(st["s"])
        alpha = torch.exp(-1.0 / self.log_tau_syn.exp())
        st["i"] = alpha * st["i"] + (1 - alpha) * current          # unit-gain synaptic filter
        v = st["v"] + st["i"]                                       # integrate-and-fire, no leak
        thr = 1.0 + self.adapt_scale * st["b"] * self.adapt_mask
        s = MaxSpike.apply(v, thr, self.max_spikes, 0.5)
        v = v - s * thr                                             # soft reset
        st["v"] = torch.clamp(v, min=self.min_vmem, max=self.max_vmem)
        st["b"] = self.alpha_adapt * st["b"] + (1 - self.alpha_adapt) * s
        st["s"] = s
        return s


class LSNN_BMI(nn.Module):
    def __init__(self, num_input_channels: int = 96, hidden_dims: Optional[List[int]] = None,
                 n_outputs: int = 2, adaptive_fraction: float = 0.9, tau_adapt_ms: float = 500.0,
                 adapt_scale: float = 1.8, tau_syn_ms=(20.0, 1300.0), out_tau_ms: float = 40.0,
                 out_stages: int = 1, readout_all: bool = False,
                 max_spikes: int = 3, rec_init_scale: float = 0.5, max_vmem: float = 50.0,
                 min_vmem: float = -1.0, velocity_lo: float = -280.56, velocity_hi: float = 316.54,
                 velocity_margin: float = 0.05):
        super().__init__()
        hidden_dims = list(hidden_dims or [256, 256])
        dims = [num_input_channels] + hidden_dims
        tau_syn_steps = [t / STEP_MS for t in tau_syn_ms]
        self.spiking = nn.ModuleList([
            LSNNLayer(dims[i], dims[i + 1], tau_syn_steps, adaptive_fraction, tau_adapt_ms / STEP_MS,
                      adapt_scale, max_spikes, rec_init_scale, max_vmem, min_vmem, seed=5 + i)
            for i in range(len(hidden_dims))])
        # readout_all: the readout reads every spiking layer, not only the last (as the speech LSNN)
        self.readout_all = bool(readout_all)
        self.readout = nn.Linear(sum(hidden_dims) if self.readout_all else hidden_dims[-1], n_outputs)
        nn.init.kaiming_uniform_(self.readout.weight)
        # start at the scaled value of zero velocity (the dataloader's scaling of v = 0)
        neutral = velocity_margin - velocity_lo * (1 - 2 * velocity_margin) / (velocity_hi - velocity_lo)
        nn.init.constant_(self.readout.bias, float(neutral))
        if out_stages < 1:
            raise ValueError(f"out_stages must be >= 1, got {out_stages}")
        self.out_log_tau = nn.Parameter(torch.full((out_stages, n_outputs), math.log(out_tau_ms / STEP_MS)))
        # what train_snn.py reads: model.layers[0] (channel lasso), layer widths, neuron count
        self.layers = nn.ModuleList([self.spiking[0].ff])
        self.layer_widths = list(hidden_dims)
        self.total_neuron_units = sum(hidden_dims)

    def forward(self, x_total: torch.Tensor, count_ops: bool = False, reset_state: bool = True):
        T, N = x_total.shape[:2]
        device = x_total.device
        a_out = torch.exp(-1.0 / self.out_log_tau.exp())
        carried = getattr(self, "_carried", None)
        if reset_state or carried is None or carried[1][0].shape[0] != N:
            states = [layer.init_state(N, device) for layer in self.spiking]
            ys = [self.readout.bias.expand(N, -1).clone() for _ in range(a_out.shape[0])]
        else:
            states, ys = carried
        drive = self.spiking[0].ff(x_total)                          # first layer's input, all steps at once
        if count_ops:
            mac = int(torch.count_nonzero(x_total).item()) * self.spiking[0].width
            acc, elementwise = 0.0, 0
            n_adapt = [int(layer.adapt_mask.sum().item()) for layer in self.spiking]
        self.layer_spikes = [torch.zeros((), device=device) for _ in self.spiking]
        preds, final_spikes, total_spikes = [], torch.zeros(1, device=device), torch.zeros(1, device=device)
        for t in range(T):
            current = drive[t]
            layer_out = []
            for li, (layer, st) in enumerate(zip(self.spiking, states)):
                if count_ops:
                    acc += float(st["s"].sum().item()) * layer.width                       # recurrent synapses
                    if li > 0:
                        acc += float(s.sum().item()) * layer.width                         # from the layer below
                    mac += N * (layer.width + 2 * n_adapt[li])     # synaptic filter, adaptation decay + update
                    elementwise += 2 * N * layer.width             # integrate + threshold/reset
                if li > 0:
                    current = layer.ff(s)
                s = layer.step(current, st)
                layer_out.append(s)
                n = s.sum()
                self.layer_spikes[li] = self.layer_spikes[li] + n
                total_spikes = total_spikes + n
            final_spikes = final_spikes + s.sum()
            read = torch.cat(layer_out, -1) if self.readout_all else s
            if count_ops:
                acc += float(read.sum().item()) * self.readout.out_features              # readout synapses
                mac += N * 2 * self.readout.out_features * len(ys)                       # leaky readout stages
            stage_in = self.readout(read)                                                # leaky stages, never spike
            for k in range(len(ys)):
                ys[k] = a_out[k] * ys[k] + (1 - a_out[k]) * stage_in
                stage_in = ys[k]
            preds.append(ys[-1])
        self._carried = ([{k: v.detach() for k, v in st.items()} for st in states], [y.detach() for y in ys])
        if count_ops:
            return torch.stack(preds, 0), final_spikes, total_spikes, {
                'mac': mac, 'acc': acc, 'elementwise': elementwise}
        return torch.stack(preds, 0), final_spikes, total_spikes


def create_model(num_input_channels: int = 96, hidden_dims: Optional[List[int]] = None, **kwargs) -> LSNN_BMI:
    """Factory used by train_snn.py --arch lsnn; kwargs are LSNN_BMI's options."""
    return LSNN_BMI(num_input_channels=num_input_channels, hidden_dims=hidden_dims, **kwargs)


def load_model_weights(model: LSNN_BMI, state_dict: dict, neuron_type: str = None,
                       source_description: str = "checkpoint") -> None:
    """train_snn.py --init-weights-from: the LSNN keeps no lazily shaped state buffers, so a strict load."""
    model.load_state_dict(state_dict, strict=True)
