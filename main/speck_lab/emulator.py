"""
Host emulation of the deployed network as Speck runs it, one input event at a time.

What the chip does differently from training (probe_speck.py, measured on a
Speck2f devkit): it has no timesteps. Each input event updates the neurons
it projects to at once; a neuron whose membrane reaches threshold fires one
spike (never several for one event) and is reset (hard: to 0; soft: minus
the threshold) before the next event arrives; the membrane's lower bound is
enforced after every event. The spikes it emits are events for the next
layer. So within one 4 ms timestep the result depends on the order in which
excitatory and inhibitory events arrive, which training never sees.

This emulates that with the chip's integer weights and thresholds
(quantize(): sinabs' own discretization of each Linear + IAF pair, as
DynapcnnNetwork applies it).

In a feedforward network a layer's output depends only on the order of the
events it receives, not on how the layers interleave in time, so only the
order within a timestep matters. The probe could not settle it, so it is an
option, to be checked against the real chip with lab.py validate:

    shuffle 'none' (default): input events in channel order (as
            ChipFactory.raster_to_events writes them), each layer's spikes
            in neuron order; 'input': input events shuffled; 'all': every
            layer's events shuffled.

Membranes persist across timesteps (IAF, no leak) and are zeroed at the start
of each trial (SpeckDevkit.reset()).
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn


def _sinabs_discretize():
    """sinabs.backend.dynapcnn.discretize, loaded on its own: importing the
    package itself needs samna, which machines without a devkit lack."""
    import sinabs
    path = os.path.join(os.path.dirname(sinabs.__file__), 'backend', 'dynapcnn', 'discretize.py')
    spec = importlib.util.spec_from_file_location('_sinabs_dynapcnn_discretize', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantizedLayer:
    """One Linear + IAF pair as integers: weight (out, in), threshold,
    lower bound, reset ('hard' or 'soft')."""

    def __init__(self, weight, threshold, min_v, reset):
        self.weight = np.asarray(weight, dtype=np.float64)
        self.threshold = float(threshold)
        self.min_v = None if min_v is None else float(min_v)
        self.reset = reset


def quantize(snn_seq):
    """The flattened network (Linear, IAF, Linear, IAF, ...) as QuantizedLayers,
    discretized as for the chip. Also returns the matching nn.Sequential of
    1x1 Conv2d + IAF layers, which steps the quantized network one timestep
    per call like training does."""
    from sinabs.activation import MembraneReset
    disc = _sinabs_discretize()
    modules = list(snn_seq)
    layers, seq = [], []
    for linear, neuron in zip(modules[0::2], modules[1::2]):
        conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(linear.weight[:, :, None, None])
        conv_q, neuron_q = disc.discretize_conv_spike(conv, neuron, to_int=False)
        min_v = getattr(neuron_q, 'min_v_mem', None)
        layers.append(QuantizedLayer(
            conv_q.weight.detach().reshape(conv_q.out_channels, -1).numpy(),
            float(torch.as_tensor(neuron_q.spike_threshold).reshape(-1)[0]),
            None if min_v is None else float(torch.as_tensor(min_v).reshape(-1)[0]),
            'hard' if isinstance(neuron_q.reset_fn, MembraneReset) else 'soft'))
        seq += [conv_q, neuron_q]
    return layers, nn.Sequential(*seq)


class ChipEmulator:
    def __init__(self, layers, shuffle='none', seed=0):
        if shuffle not in ('none', 'input', 'all'):
            raise ValueError(f"shuffle must be 'none', 'input' or 'all', got {shuffle!r}")
        self.layers, self.shuffle = layers, shuffle
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.v = [np.zeros(layer.weight.shape[0]) for layer in self.layers]

    def _event(self, k, source):
        """One event from `source` (an index into layer k's inputs) into layer
        k: update, fire at most once, reset. Returns the neurons that fired,
        in index order."""
        layer, v = self.layers[k], self.v[k]
        v += layer.weight[:, source]
        if layer.min_v is not None:
            np.maximum(v, layer.min_v, out=v)
        fired = np.flatnonzero(v >= layer.threshold)
        if fired.size:
            if layer.reset == 'hard':
                v[fired] = 0.0
            else:
                v[fired] -= layer.threshold
        return fired

    def step(self, input_events):
        """One timestep: input_events lists input channels, one entry per
        event (a channel with n spikes appears n times). Returns spike counts
        per neuron of every layer."""
        counts = [np.zeros(layer.weight.shape[0]) for layer in self.layers]
        events = np.asarray(input_events, dtype=int)
        for k in range(len(self.layers)):
            if self.shuffle == 'all' or (self.shuffle == 'input' and k == 0):
                events = self.rng.permutation(events)
            emitted = []
            for e in events:
                fired = self._event(k, e)
                counts[k][fired] += 1
                emitted.extend(fired.tolist())
            events = np.asarray(emitted, dtype=int)
        return counts

    def run(self, input_spikes):
        """(C, T) input spike counts from rest -> [(T, n) spike counts per layer]."""
        self.reset()
        x = np.rint(np.asarray(input_spikes)).astype(int)
        out = [np.zeros((x.shape[1], layer.weight.shape[0])) for layer in self.layers]
        for t in range(x.shape[1]):
            column = x[:, t]
            channels = np.flatnonzero(column)
            events = np.repeat(channels, column[channels])
            for k, c in enumerate(self.step(events)):
                out[k][t] = c
        return out
