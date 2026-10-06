"""
Run the SNN on a Speck2f devkit (the 'speck' decoder of test_all_decoders.py).

The trained model's Linear/IAF stack is flattened to an nn.Sequential,
checked against model() in software, converted with DynapcnnNetwork
(weights discretized to the chip's precision) and deployed. Each timestep's
input is written to the chip as events; after --speck_wait_time the output
layer's spikes are counted per feature and passed through the model's own
EMA cascade and population decode on the host, since the chip runs only the
neuron layers.

Latency is wall time per timestep of that loop (including the wait), and
power is the devkit PowerMonitor's mean over the same loop, so energy per
sample = power x latency.

Each layer's reset after a spike follows training: sinabs's make_config sets
return_to_zero for a hard reset and subtracts the threshold for a soft one.
Unlike training, which sums a timestep's input and then fires, the chip
updates a neuron after every input event and fires at most once per event
(probe_speck.py measures this), so a hard reset discards the charge above
threshold at every spike rather than once per timestep.

run_discretized() steps the discretized network (8-bit weights, integer
thresholds) one timestep at a time, like training, for diagnose_speck.py;
run_layers() does the same for either network but returns every neuron
layer's spikes. A SpeckDevkit opened with monitor_all=True streams every
layer's spikes from the chip as well (for diagnose_speck.py's layer
activity figure; the extra traffic makes its latency and power
unrepresentative).

samna and the dynapcnn backend are imported only when a chip run is
requested, so machines without a devkit never need them.
"""

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from decoder_eval import BASE_NPERSEG, _load_pickle, snn_test_files, unscale_velocity

POWER_SAMPLE_RATE_HZ = 100
RESET_SETTLE_S = 1.0     # after writing zeroed membrane values to a layer


# --------------------------------------------------------------------------- #
# Host-side model pieces
# --------------------------------------------------------------------------- #

def flatten_snn(model):
    """The model's layers as a plain nn.Sequential of Linear and sinabs
    neuron layers (the neuron wrappers unwrapped), as DynapcnnNetwork
    expects."""
    return nn.Sequential(*[getattr(m, 'neuron', m) for m in model.layers])


def ema_cascade_update(stages, x, decay):
    """One timestep of the model's EMA readout: stage 0 is fed the output
    spikes, each later stage the previous stage; returns the last stage."""
    stage_input = x
    for i in range(len(stages)):
        stages[i] = decay * stages[i] + stage_input
        stage_input = stages[i]
    return stages[-1]


def decode_spike_counts(model, spike_counts):
    """(T, 2 * n_bins) output spike counts -> (T, 2) scaled velocity through
    the model's EMA cascade and decode_output()."""
    stages = [torch.zeros(1, 2 * model.n_bins) for _ in range(model.temporal_decay_stages)]
    decay = model.temporal_decay
    counts = torch.as_tensor(spike_counts, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([model.decode_output(ema_cascade_update(stages, counts[t:t + 1], decay))
                          for t in range(len(counts))]).numpy()


def discretize(snn_seq, n_inputs):
    """The DynapcnnNetwork deployed to the chip (weights and thresholds
    quantized to the chip's integer ranges)."""
    from sinabs.backend.dynapcnn import DynapcnnNetwork
    return DynapcnnNetwork(nn.Sequential(nn.Flatten(), *snn_seq), input_shape=(n_inputs, 1, 1),
                           discretize=True, dvs_input=False)


def discretized_sequential(network):
    """A DynapcnnNetwork's quantized layers as a plain nn.Sequential of
    Conv2d and IAF layers."""
    if hasattr(network, 'sequence'):                     # sinabs < 3
        pairs = [(l.conv_layer, l.spk_layer) for l in network.sequence if hasattr(l, 'conv_layer')]
    else:
        layers = network._dynapcnn_module._dynapcnn_layers
        pairs = [(layers[k].conv, layers[k].spk) for k in sorted(layers, key=int)]
    return nn.Sequential(*[m for pair in pairs for m in pair])


def run_float(snn_seq, input_spikes):
    """(C, T) input -> (T, n_out) output spikes of the flattened network."""
    import sinabs
    x = torch.from_numpy(input_spikes.T.astype(np.float32))
    sinabs.reset_states(snn_seq)
    with torch.no_grad():
        return torch.cat([snn_seq(x[t:t + 1]) for t in range(len(x))]).numpy()


def run_discretized(disc_seq, input_spikes):
    """(C, T) input -> (T, n_out) output spikes of the quantized network,
    one timestep per call (state carried between calls, as in training)."""
    import sinabs
    x = torch.from_numpy(input_spikes.T.astype(np.float32))[:, :, None, None]
    sinabs.reset_states(disc_seq)
    with torch.no_grad():
        return torch.cat([disc_seq(x[t:t + 1]).reshape(1, -1) for t in range(len(x))]).numpy()


def run_layers(seq, input_spikes):
    """(C, T) input -> [(T, n) spikes of each neuron layer of seq], in order,
    one timestep per call; seq is the flattened (Linear) or the discretized
    (Conv2d) network."""
    import sinabs
    conv = any(isinstance(m, nn.Conv2d) for m in seq)
    x = torch.from_numpy(input_spikes.T.astype(np.float32))
    if conv:
        x = x[:, :, None, None]
    sinabs.reset_states(seq)
    steps = []
    with torch.no_grad():
        for t in range(len(x)):
            out, spikes = x[t:t + 1], []
            for module in seq:
                out = module(out)
                if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Flatten)):
                    spikes.append(out.reshape(-1).numpy())
            steps.append(spikes)
    return [np.stack(layer) for layer in zip(*steps)]


def check_deployable(model, checkpoint):
    args = checkpoint['args']
    if args.get('neuron_type') != 'iaf':
        raise ValueError(f"Only IAF networks run on Speck; this checkpoint has "
                         f"neuron_type={args.get('neuron_type')!r}")
    if args.get('use_spikingjelly'):
        raise ValueError("Only sinabs networks run on Speck; this checkpoint uses SpikingJelly")
    if any(k.endswith('.tau_syn') for k in checkpoint['model_state_dict']):
        raise ValueError("Speck has no synaptic current stage; this checkpoint was trained with tau_syn")


def cross_check_flattened(model, snn_seq, input_spikes, tol=1e-4):
    """Run one trial (C, T) through snn_seq + host decode and model(); they
    must agree, or the network deployed to the chip is not the trained one."""
    import sinabs
    x = torch.from_numpy(input_spikes.T.astype(np.float32))           # (T, C)
    sinabs.reset_states(snn_seq)
    with torch.no_grad():
        spikes = torch.cat([snn_seq(x[t:t + 1]) for t in range(len(x))])
        flat = decode_spike_counts(model, spikes)
        ref = model(x[:, None, :])[0].squeeze(1).numpy()
    sinabs.reset_states(snn_seq)
    max_diff = float(np.abs(flat - ref).max())
    if max_diff > tol:
        raise RuntimeError(f"Flattened network differs from model() by {max_diff:.2e} "
                           f"(scaled velocity); not deploying it")
    print(f"  [speck] flattened network matches model() (max diff {max_diff:.1e})")


# --------------------------------------------------------------------------- #
# Devkit
# --------------------------------------------------------------------------- #

def measure_chip_power(power_events, loop_s, sample_rate_hz=POWER_SAMPLE_RATE_HZ):
    """(power_w, sample_count_ok): the sum over PowerMonitor channels of each
    channel's mean power. The events' own timestamps arrive in bursts, so
    they are not integrated; instead the sample count is checked against
    sample_rate_hz x loop_s per channel (within 20%)."""
    if not power_events:
        return 0.0, False
    by_channel = defaultdict(list)
    for ev in power_events:
        by_channel[ev.channel].append(float(ev.value))
    power_w = float(sum(np.mean(v) for v in by_channel.values()))
    expected = sample_rate_hz * loop_s * len(by_channel)
    return power_w, abs(len(power_events) - expected) <= 0.2 * expected


class SpeckDevkit:
    """A converted network deployed on a Speck2f devkit."""

    def __init__(self, snn_seq, n_inputs, devkit='speck2fdevkit:0', wait_time=0.001, raster_dt=0.1,
                 monitor_all=False):
        import samna
        import sinabs.backend.dynapcnn.io as sio
        from sinabs.backend.dynapcnn.chip_factory import ChipFactory

        self.samna, self.wait_time, self.raster_dt = samna, wait_time, raster_dt
        self.monitor_all, self.layer_counts = monitor_all, None
        self.network = discretize(snn_seq, n_inputs)
        self.layer_sizes = [m.out_channels for m in discretized_sequential(self.network)
                            if isinstance(m, nn.Conv2d)]
        self.param_count = sum(p.numel() for m in self.network.modules() if isinstance(m, nn.Conv2d)
                               for p in (m.weight, m.bias) if p is not None)

        self.device = sio.open_device(devkit)
        self.chip_factory = ChipFactory(devkit.split(':')[0])
        self.stop_watch = self.device.get_stop_watch()
        self.power_monitor = self.device.get_power_monitor()
        self.power_sink = samna.BasicSinkNode_unifirm_modules_events_measurement()
        self.input_source = samna.BasicSourceNode_speck2f_event_input_event()
        self.output_sink = samna.BasicSinkNode_speck2f_event_output_event()
        self.graph = samna.graph.EventFilterGraph()
        self.graph.sequential([self.input_source, self.device.get_model_sink_node()])
        self.graph.sequential([self.device.get_model_source_node(), self.output_sink])
        self.graph.sequential([self.power_monitor.get_source_node(), self.power_sink])
        self.graph.start()
        self.stop_watch.set_enable_value(True)

        config = self.network.make_config(device=devkit)       # reset from each layer's reset_fn
        ordering = self.network.chip_layers_ordering   # list, or {layer: core} in sinabs >= 3
        self.cores = [ordering[k] for k in sorted(ordering)] if isinstance(ordering, dict) else list(ordering)
        config.dvs_layer.pass_sensor_events = False
        for core in self.cores:
            # Normally only the output layer is read; monitoring the hidden
            # layers streams all of their spikes to the host as well.
            config.cnn_layers[core].monitor_enable = monitor_all or core == self.cores[-1]
        self.device.get_model().apply_configuration(config)
        self.resets = ['hard' if config.cnn_layers[c].return_to_zero else 'soft' for c in self.cores]
        print(f"  [speck] {devkit}: chip layers {self.cores}, reset {'/'.join(self.resets)}, "
              f"{self.param_count:,} parameters deployed")

    def reset(self):
        """Zero every membrane potential and drop pending output and power events."""
        self.output_sink.get_events()
        builder = self.chip_factory.get_config_builder()
        module = builder.get_samna_module()
        for core in self.cores:
            events = []
            for address in range(builder.get_constraints()[core].neuron_memory):
                ev = module.event.WriteNeuronValue()
                ev.address, ev.layer, ev.neuron_state = address, core, 0
                events.append(ev)
            source = builder.get_input_buffer()
            graph = self.samna.graph.EventFilterGraph()
            graph.sequential([source, self.device.get_model().get_sink_node()])
            graph.start()
            source.write(events)
            time.sleep(RESET_SETTLE_S)
            graph.stop()
        self.power_sink.get_events()

    def run(self, input_spikes, n_outputs):
        """Feed one trial (C, T) a timestep at a time. Returns (output spike
        counts (T, n_outputs), loop seconds, mean power in W, input events
        sent). Output spikes still in flight after the wait are counted in
        a later timestep. With monitor_all, self.layer_counts is then the
        list of every layer's spike counts (T, layer size), in order."""
        frames = torch.from_numpy(input_spikes.T.astype(np.float32))[:, None, :, None, None]
        spike_type, last_core = self.samna.speck2f.event.Spike, self.cores[-1]
        counts = np.zeros((len(frames), n_outputs))
        layer_counts = [np.zeros((len(frames), n)) for n in self.layer_sizes] if self.monitor_all else None
        n_input_events = 0
        self.power_monitor.start_auto_power_measurement(POWER_SAMPLE_RATE_HZ)
        start = time.perf_counter()
        for t, frame in enumerate(frames):
            if frame.any():
                events = self.chip_factory.raster_to_events(frame, self.cores[0], dt=self.raster_dt)
                n_input_events += len(events)
                self.stop_watch.start(reset=True)
                self.input_source.write(events)
            time.sleep(self.wait_time)
            spikes = [ev for ev in self.output_sink.get_events() if isinstance(ev, spike_type)]
            features = [ev.feature for ev in spikes if ev.layer == last_core]
            counts[t] = np.bincount(features, minlength=n_outputs)[:n_outputs]
            if layer_counts is not None:
                for core, layer in zip(self.cores, layer_counts):
                    features = [ev.feature for ev in spikes if ev.layer == core]
                    layer[t] = np.bincount(features, minlength=layer.shape[1])[:layer.shape[1]]
        loop_s = time.perf_counter() - start
        self.power_monitor.stop_auto_power_measurement()
        power_events = self.power_sink.get_events()
        power_w, count_ok = measure_chip_power(power_events, loop_s)
        self.layer_counts = layer_counts
        if not count_ok:
            print(f"  WARNING: {len(power_events)} power samples over {loop_s:.2f} s is not what "
                  f"{POWER_SAMPLE_RATE_HZ} Hz per channel predicts; this trial's power may be unreliable")
        return counts, loop_s, power_w, n_input_events

    def close(self):
        self.graph.stop()


# --------------------------------------------------------------------------- #
# Test set
# --------------------------------------------------------------------------- #

def open_speck(model, checkpoint, snn_dataset_path, devkit, wait_time, raster_dt, monitor_all=False):
    """Check the checkpoint can be deployed, cross-check the flattened
    network on the first test trial, and deploy it."""
    check_deployable(model, checkpoint)
    snn_seq = flatten_snn(model)
    snn_seq.eval()
    cross_check_flattened(model, snn_seq, _load_pickle(snn_test_files(snn_dataset_path)[0])['input_spikes'])
    return SpeckDevkit(snn_seq, model.layers[0].in_features, devkit, wait_time, raster_dt, monitor_all)


def predict_speck_test_set(model, velocity_scale, snn_dataset_path, device, experiment,
                           continuous_stream=False):
    """The chip counterpart of decoder_eval.predict_snn_test_set(): the same
    trials, the same dropped first window, so its output aligns the same way.

    Returns (pred (n, 2), chip, output spike counts (n, 2 * n_bins)) with
    chip = {latency_s, energy_j (per sample), power_w, param_count,
    n_timesteps, output_spikes_per_step, input_events_sent,
    input_spikes, wait_time_s, raster_dt}; input_events_sent should equal
    input_spikes (the summed input counts)."""
    files = snn_test_files(snn_dataset_path)
    continuous = continuous_stream and experiment == 'hkm'
    n_outputs = 2 * model.n_bins
    preds, kept_counts, loop_s, energy_j, n_timed, n_spikes = [], [], 0.0, 0.0, 0, 0.0
    n_events, n_input = 0, 0
    for i, path in enumerate(files):
        if i == 0 and len(files) > 1 and not continuous:
            continue                                  # dropped anyway; skip the chip time
        if i == 0 or not continuous:
            device.reset()
        spikes = _load_pickle(path)['input_spikes']
        counts, seconds, power_w, events = device.run(spikes, n_outputs)
        n_events, n_input = n_events + events, n_input + int(np.round(spikes).sum())
        print(f"  [speck] trial {i + 1}/{len(files)}: {spikes.shape[1]} timesteps, "
              f"{seconds * 1000 / spikes.shape[1]:.3f} ms/timestep, {power_w * 1000:.3f} mW")
        loop_s, energy_j, n_timed = loop_s + seconds, energy_j + power_w * seconds, n_timed + len(counts)
        n_spikes += counts.sum()
        y_pred = unscale_velocity(decode_spike_counts(model, counts), velocity_scale)
        if len(files) == 1:
            preds.append(y_pred[BASE_NPERSEG:])
            kept_counts.append(counts[BASE_NPERSEG:])
        elif i > 0:
            preds.append(y_pred)
            kept_counts.append(counts)
    if n_events != n_input:
        print(f"  WARNING: {n_events} input events sent to the chip for {n_input} input spikes")
    chip = {'latency_s': loop_s / n_timed, 'energy_j': energy_j / n_timed,
            'power_w': energy_j / loop_s if loop_s > 0 else None,
            'param_count': int(device.param_count), 'n_timesteps': n_timed,
            'output_spikes_per_step': float(n_spikes / n_timed),
            'input_events_sent': int(n_events), 'input_spikes': int(n_input),
            'wait_time_s': device.wait_time, 'raster_dt': device.raster_dt}
    return np.concatenate(preds, axis=0), chip, np.concatenate(kept_counts, axis=0)
