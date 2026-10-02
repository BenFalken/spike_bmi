"""
Operation-count energy ESTIMATES for each decoder, independent of the machine
it runs on (energy_meter.py MEASURES energy instead).

For one test-set evaluation, each estimate_ops_*() counts effective
multiply-accumulates (MACs, one per nonzero input x weight, since zero inputs
can be skipped) and accumulates (ACCs, for binary spike inputs, which need no
multiply), derives memory accesses (MAC = 4, ACC = 3; Liao et al.), and converts
to joules with Horowitz (ISSCC 2014) 45 nm FP32 figures: MAC 4.6 pJ, ACC 0.9 pJ,
memory access 5 pJ (on-chip SRAM) to 640 pJ (off-chip DRAM). Memory placement
depends on the hardware, so total energy is reported as a range,
energy_total_j_low (all SRAM) to energy_total_j_high (all DRAM), both in total
and per sample.

KF counts its per-step matrix algebra, with the c x c inversion as c^3 MACs (an
approximation; it dominates for ~96 channels). The SNN's counts come from
SNN_Speck.forward(count_ops=True) and go through finalize_snn_ops().
"""

import numpy as np

# Horowitz (2014), 45 nm, FP32
ENERGY_PER_MAC_PJ = 3.7 + 0.9          # multiply + add
ENERGY_PER_ACC_PJ = 0.9                # add only
ENERGY_PER_MEM_ACCESS_SRAM_PJ = 5.0
ENERGY_PER_MEM_ACCESS_DRAM_PJ = 640.0
_PJ_TO_J = 1e-12


def memory_accesses_from_ops(effective_macs, effective_accs):
    """MAC = 3 loads + 1 store; ACC = 2 loads + 1 store (Liao et al.)."""
    return effective_macs * 4 + effective_accs * 3


def estimate_energy_joules(effective_macs, effective_accs, memory_accesses):
    compute = (effective_macs * ENERGY_PER_MAC_PJ + effective_accs * ENERGY_PER_ACC_PJ) * _PJ_TO_J
    sram = memory_accesses * ENERGY_PER_MEM_ACCESS_SRAM_PJ * _PJ_TO_J
    dram = memory_accesses * ENERGY_PER_MEM_ACCESS_DRAM_PJ * _PJ_TO_J
    return {'energy_compute_j': compute, 'energy_memory_j_sram': sram, 'energy_memory_j_dram': dram,
            'energy_total_j_low': compute + sram, 'energy_total_j_high': compute + dram}


def _finalize(effective_macs, effective_accs, elementwise_ops=0, n_samples=None):
    """Totals plus per-sample values. elementwise_ops (e.g. matrix additions)
    are reported but not included in the energy."""
    mem = memory_accesses_from_ops(effective_macs, effective_accs)
    energy = estimate_energy_joules(effective_macs, effective_accs, mem)
    result = {'effective_macs': effective_macs, 'effective_accs': effective_accs,
              'elementwise_ops': elementwise_ops, 'memory_accesses': mem, **energy}
    if n_samples:
        result['per_sample'] = {
            'effective_macs': effective_macs / n_samples,
            'effective_accs': effective_accs / n_samples,
            'memory_accesses': mem / n_samples,
            'energy_total_j_low': energy['energy_total_j_low'] / n_samples,
            'energy_total_j_high': energy['energy_total_j_high'] / n_samples,
        }
    return result


def estimate_ops_kf(model, n_timesteps):
    """model: fitted KalmanDecoder (model.model = [A, W, H, Q]); n_timesteps:
    recursion steps run (predict() runs n_test_samples - 1). With state size s
    and c channels, one step costs (in MACs):
        P_m = A P A^T: 2 s^3      state_m = A state: s^2
        H P_m H^T:  c s^2 + c^2 s      inv(c x c): c^3 (Gauss-Jordan, approximate)
        K = P_m H^T inv:  s^2 c + s c^2      P = (I - K H) P_m:  s^2 c + s^3
        H state_m, K residual:  2 c s"""
    A, _, H, _ = (np.asarray(m) for m in model.model)
    s, c = A.shape[0], H.shape[0]
    macs_per_step = (2 * s**3 + s**2 + c * s**2 + c**2 * s + c**3
                     + s**2 * c + s * c**2 + s**2 * c + s**3 + c * s + s * c)
    elementwise_per_step = s**2 + c**2 + s**2 + c + s        # +W, +Q, I - KH, residual, state update
    return _finalize(macs_per_step * n_timesteps, 0,
                     elementwise_ops=elementwise_per_step * n_timesteps, n_samples=n_timesteps)


def estimate_ops_wf(model, X):
    """model: fitted WienerDecoder; X: the flattened tap-delay input it predicts
    from. Each nonzero input feature costs one MAC per output."""
    X = np.asarray(X)
    n_outputs = np.atleast_2d(model.model.coef_).shape[0]
    return _finalize(int(np.count_nonzero(X)) * n_outputs, 0, n_samples=X.shape[0])


def _run_layers_capturing(model, X, capture_types):
    """Run X through a Keras Sequential one layer at a time, yielding
    (layer, its input) for layers whose class name is in capture_types
    (Dropout is a no-op at inference)."""
    import tensorflow as tf
    x = tf.convert_to_tensor(X)
    for layer in model.layers:
        if layer.__class__.__name__ in capture_types:
            yield layer, np.asarray(x)
        x = layer(x)


def _dense_macs(layer, layer_input):
    return int(np.count_nonzero(layer_input)) * layer.get_weights()[0].shape[1]


def estimate_ops_lstm(model, X):
    """X: (n_samples, timesteps, channels). Per timestep, each nonzero input
    costs one MAC per gate unit (4 x units), and the recurrent state costs
    units x 4 units MACs (treated as dense); plus the Dense output layer."""
    X = np.asarray(X)
    n_samples, timesteps, _ = X.shape
    total_macs = 0
    for layer, layer_input in _run_layers_capturing(model, X, {'LSTM', 'Dense'}):
        if layer.__class__.__name__ == 'LSTM':
            kernel, recurrent_kernel, _ = layer.get_weights()
            units, gate_width = recurrent_kernel.shape[0], kernel.shape[1]
            total_macs += int(np.count_nonzero(layer_input, axis=-1).sum()) * gate_width
            total_macs += n_samples * timesteps * units * gate_width
        else:
            total_macs += _dense_macs(layer, layer_input)
    return _finalize(total_macs, 0, n_samples=n_samples)


def estimate_ops_qrnn(model, X):
    """X: (n_samples, timesteps, channels). The QRNN's causal convolution
    (window_size taps, left-padded) uses input timestep i in
    min(T - i, window_size) windows; each use of a nonzero input costs one MAC
    per output channel (3 x units); plus the Dense output layer."""
    X = np.asarray(X)
    total_macs = 0
    for layer, layer_input in _run_layers_capturing(model, X, {'QRNN', 'Dense'}):
        if layer.__class__.__name__ == 'QRNN':
            kernel = layer.get_weights()[0]                 # (window_size, 1, input_dim, 3 * units)
            window_size, out_channels = kernel.shape[0], kernel.shape[-1]
            T = layer_input.shape[1]
            uses = np.minimum(T - np.arange(T), window_size)
            weighted_nnz = int((np.count_nonzero(layer_input, axis=-1) * uses[np.newaxis, :]).sum())
            total_macs += out_channels * weighted_nnz
        else:
            total_macs += _dense_macs(layer, layer_input)
    return _finalize(total_macs, 0, n_samples=X.shape[0])


def finalize_snn_ops(mac, acc, elementwise, n_samples):
    """Convert SNN_Speck.forward(count_ops=True) counts to the same result
    format; n_samples = timesteps x batch size of the call(s)."""
    return _finalize(mac, acc, elementwise_ops=elementwise, n_samples=n_samples)
