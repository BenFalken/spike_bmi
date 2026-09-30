"""
Measure how a Speck2f neuron integrates and fires, with a one-neuron network
on the devkit (run from main/inference, devkit connected).

Each case sends a pattern of input events (events per timestep) to one IAF
neuron with threshold 1.0 and a given input weight, under hard reset
(return to zero) or soft reset (subtract the threshold), and prints the
output spikes per timestep next to what the training model (sum the
timestep's input, fire floor(v / threshold) spikes, then reset) would give.

Results on a Speck2f devkit (sinabs 3.1, weights 127, thresholds scaled):
the first input event after a membrane reset is lost; afterwards the neuron
is updated after every input event and fires at most once per event, with
the configured reset applied after each spike.
"""

import numpy as np
import torch.nn as nn
import sinabs.backend.dynapcnn.io as sio
import sinabs.layers as sl
from sinabs.activation import MembraneReset, MembraneSubtract, MultiSpike

import speck

DEVKIT = 'speck2fdevkit:0'
CASES = [  # (weight in thresholds, input events per timestep)
    (0.7, [3, 0, 0]),     # v reaches 2.1 in one timestep
    (0.7, [1, 0, 0]),     # below threshold
    (0.9, [4, 0, 0]),     # v reaches 3.6 in one timestep
    (2.5, [1, 0, 0]),     # a single event worth 2.5 thresholds
    (2.5, [2, 0, 0]),     # the same after a (possibly lost) first event
    (1.2, [1, 1, 1, 1]),  # every event can fire on its own: shows which events arrive
    (1.2, [0, 1, 0, 1]),
    (1.2, [2, 2, 2, 2]),
]


def training_model(weight, pattern, hard):
    """Output spikes per timestep of the trained neuron (MultiSpike)."""
    v, out = 0.0, []
    for n in pattern:
        v += weight * n
        spikes = int(v // 1.0) if v >= 1.0 else 0
        v = 0.0 if (hard and spikes) else v - spikes
        out.append(spikes)
    return out


def probe(weight, pattern, hard):
    linear = nn.Linear(1, 1, bias=False)
    linear.weight.data[:] = weight
    neuron = sl.IAFSqueeze(spike_threshold=1.0, spike_fn=MultiSpike, min_v_mem=-1.0, batch_size=1,
                           reset_fn=MembraneReset() if hard else MembraneSubtract())
    device = speck.SpeckDevkit(nn.Sequential(linear, neuron), n_inputs=1, wait_time=0.05, devkit=DEVKIT)
    try:
        layer = speck.discretized_sequential(device.network)
        chip = f"weight {int(layer[0].weight.item())}, threshold {int(layer[1].spike_threshold.item())}"
        device.reset()
        counts = device.run(np.array([pattern], dtype=float), 1)[0][:, 0]
    finally:
        device.close()
        try:
            sio.close_device(DEVKIT)    # so the next case can open it again
        except Exception:
            pass
    return counts.astype(int).tolist(), chip, device.resets[0]


if __name__ == '__main__':
    for hard in (True, False):
        for weight, pattern in CASES:
            spikes, chip, reset = probe(weight, pattern, hard)
            print(f"{reset:4s} reset | weight {weight:.1f} ({chip}) | events {pattern} -> chip {spikes}, "
                  f"training {training_model(weight, pattern, hard)}")
