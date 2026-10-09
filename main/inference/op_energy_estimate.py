"""
Hardware-agnostic energy ESTIMATION for KF/WF/MLP/LSTM/QRNN -- a
genuinely different methodology from energy_meter.py's EnergyMeter, not
a replacement for it. Complementary, meant to be used alongside it:

  - EnergyMeter MEASURES: actual joules that actually flowed through an
    actual CPU (or, for Speck, an actual chip) while a specific
    implementation ran on a specific machine. Answers "how much energy
    did THIS RUN, ON THIS MACHINE, cost."
  - This module ESTIMATES: counts effective (non-zero-exploiting)
    multiply-accumulate (MAC) and accumulate (ACC) operations per
    inference, derives memory accesses from a fixed accounting rule, and
    converts to joules using published per-operation energy figures for
    a REFERENCE process node -- deliberately WITHOUT reference to any
    specific chip actually being run on. Answers "how does this
    decoder's computational cost SCALE, independent of what it happens
    to run on."

METHODOLOGY, following the framework described in the paper excerpt this
was built from (itself following Liao et al.'s accounting convention):

  - MAC vs. ACC: a genuinely spike-driven (binary, 0/1) input needs no
    real multiplication against a weight -- weight*1 = weight -- so it's
    counted as an ACCUMULATE (just an add). A continuous-valued input
    (a real spike COUNT, or a continuous hidden/cell state) needs an
    actual multiply, so it's counted as a MULTIPLY-ACCUMULATE. This
    project's decoders never see genuine binary spikes at the KF/WF/ANN
    level (MUA spike COUNTS are continuous-valued, if often exactly
    zero) -- so every op here is a MAC, not an ACC; the ACC path exists
    in this module for completeness/symmetry with the paper's own
    framework (and in case a genuinely spike-driven decoder is added
    later), not because anything currently in this project uses it.
  - EFFECTIVE means non-zero-exploiting: a layer's output does not
    depend on its zero-valued inputs at all (0 * anything = 0), so a
    hardware implementation that skips those contributes nothing to
    real energy use. Every estimator below counts ops against the
    ACTUAL data passed in (real nonzero fractions), not a worst-case
    dense assumption -- MUA spike-count data is often genuinely sparse,
    and that sparsity is exactly what a sparsity-exploiting accelerator
    would skip.
  - MEMORY ACCESSES, per the paper's own cited convention (Liao et al.):
    one MAC = 3 loads + 1 store = 4 accesses; one ACC = 2 loads + 1
    store = 3 accesses. See memory_accesses_from_ops().

ENERGY REFERENCE NUMBERS: Horowitz, M. (2014), "Computing's energy
problem (and what we can do about it)," ISSCC -- the same 45nm-process
energy-per-operation table cited across most of the neuromorphic/
efficient-DNN-accelerator literature (e.g. Sze et al. 2017, "Efficient
Processing of Deep Neural Networks: A Tutorial and Survey," Table II;
almost certainly also the ultimate source behind the paper excerpt's own
Liao et al. citation, since that's the standard reference everyone in
this specific subfield draws from). 32-bit FLOATING POINT figures used
throughout, matching this project's actual float32 compute:
    FP32 multiply:  3.7 pJ
    FP32 add:       0.9 pJ
    => one MAC (mult+add):  4.6 pJ  |  one ACC (add only): 0.9 pJ
    SRAM (on-chip cache) read/write: ~5 pJ per access
    DRAM (off-chip) read/write:    ~640 pJ per access

CONVERTING OP/ACCESS COUNTS TO JOULES NECESSARILY REINTRODUCES HARDWARE
DEPENDENCE -- that's exactly why the paper itself stops at op/access
COUNTS rather than a single joules number: which memory tier dominates
(on-chip SRAM vs. off-chip DRAM) depends entirely on whether the target
accelerator keeps weights on-chip, which is a real design choice, not
something derivable from the algorithm alone. Rather than pick one
number and imply false precision, every estimate below reports a RANGE:
energy_total_j_low (all memory traffic assumed on-chip SRAM -- best
case) to energy_total_j_high (all memory traffic assumed off-chip DRAM
-- worst case). Report the range, not a single point, when using this.
"""

import numpy as np

# --------------------------------------------------------------------------
# Reference energy table -- Horowitz 2014, 45nm CMOS (see module docstring)
# --------------------------------------------------------------------------
_PJ_FP32_MULT = 3.7
_PJ_FP32_ADD = 0.9
ENERGY_PER_MAC_PJ = _PJ_FP32_MULT + _PJ_FP32_ADD   # 4.6 pJ: one multiply + one add
ENERGY_PER_ACC_PJ = _PJ_FP32_ADD                    # 0.9 pJ: add only, no multiply needed
ENERGY_PER_MEM_ACCESS_SRAM_PJ = 5.0                 # on-chip cache, best case
ENERGY_PER_MEM_ACCESS_DRAM_PJ = 640.0               # off-chip DRAM, worst case

_PJ_TO_J = 1e-12


def memory_accesses_from_ops(effective_macs, effective_accs):
    """Paper's own cited convention (Liao et al.): one MAC = 3 loads + 1
    store = 4 accesses; one ACC = 2 loads + 1 store = 3 accesses
    (an ACC skips loading one of the two operands a MAC needs, since a
    spike-driven ACC's 'multiplier' input is implicitly 1, not a value
    that needs loading from memory)."""
    return effective_macs * 4 + effective_accs * 3


def estimate_energy_joules(effective_macs, effective_accs, memory_accesses):
    """Returns a dict with the compute energy (unambiguous -- doesn't
    depend on memory hierarchy) and a LOW/HIGH range for total energy
    (compute + memory), spanning best-case on-chip SRAM to worst-case
    off-chip DRAM for every memory access. See module docstring for why
    this is a range, not a single number."""
    energy_compute_j = (effective_macs * ENERGY_PER_MAC_PJ +
                         effective_accs * ENERGY_PER_ACC_PJ) * _PJ_TO_J
    energy_memory_j_sram = memory_accesses * ENERGY_PER_MEM_ACCESS_SRAM_PJ * _PJ_TO_J
    energy_memory_j_dram = memory_accesses * ENERGY_PER_MEM_ACCESS_DRAM_PJ * _PJ_TO_J
    return {
        'energy_compute_j': energy_compute_j,
        'energy_memory_j_sram': energy_memory_j_sram,
        'energy_memory_j_dram': energy_memory_j_dram,
        'energy_total_j_low': energy_compute_j + energy_memory_j_sram,
        'energy_total_j_high': energy_compute_j + energy_memory_j_dram,
    }


def _finalize(effective_macs, effective_accs, elementwise_ops=0, n_samples=None):
    """Shared tail for every estimate_ops_*() below: derive memory
    accesses + energy range, optionally normalize to PER-SAMPLE values
    too (dividing by n_samples, e.g. timesteps or trial rows) alongside
    the trial TOTAL, matching this project's established convention
    (plot_decoder_efficiency.py's own latency_s/energy_j are per-sample
    for the same reason: it's the number that's actually comparable
    across decoders/sessions with different trial lengths)."""
    mem = memory_accesses_from_ops(effective_macs, effective_accs)
    energy = estimate_energy_joules(effective_macs, effective_accs, mem)
    result = {
        'effective_macs': effective_macs,
        'effective_accs': effective_accs,
        'elementwise_ops': elementwise_ops,  # NOT included in MAC/ACC energy -- see each
        # estimator's own docstring for what's folded in here (small, non-weighted-sum ops)
        'memory_accesses': mem,
        **energy,
    }
    if n_samples:
        result['per_sample'] = {
            'effective_macs': effective_macs / n_samples,
            'effective_accs': effective_accs / n_samples,
            'memory_accesses': mem / n_samples,
            'energy_total_j_low': energy['energy_total_j_low'] / n_samples,
            'energy_total_j_high': energy['energy_total_j_high'] / n_samples,
        }
    return result


# --------------------------------------------------------------------------
# KF -- dense recursive linear algebra, NO sparsity to exploit (state and
# covariance matrices are continuous-valued throughout the recursion, not
# spike-driven) -- every operation counted below is a MAC.
# --------------------------------------------------------------------------

def estimate_ops_kf(model, n_timesteps):
    """model: a fitted bmi.decoders.KalmanDecoder (model.model = [A, W, H, Q]).
    n_timesteps: number of recursion steps actually run -- KalmanDecoder.predict()
    loops range(Z.shape[1]-1), i.e. (n_test_samples - 1) steps for an
    n_test_samples-row test call; pass THAT number, not n_test_samples itself.

    Walks KalmanDecoder.predict()'s own update equations directly (see
    decoders.py), term by term, using the ACTUAL fitted A/H matrix shapes
    -- s = state dimensionality (A is s x s), c = measurement/channel
    dimensionality (H is c x s). Every matmul is counted via the standard
    dense (m x k)@(k x n) -> m*k*n MACs formula. Matrix inversion of the
    c x c innovation covariance is treated as its MAC-equivalent cost via
    the standard Gauss-Jordan flop count (~2*n^3 FLOPs for an n x n
    inverse -- see Golub & Van Loan, "Matrix Computations" -- halved to
    MAC-equivalents since one MAC = 2 FLOPs=1 mult+1 add): explicitly an
    APPROXIMATION, flagged as such, since matrix inversion isn't
    literally decomposable into weighted-sum MACs the way a neural
    network layer is -- but it dominates the total regardless of exactly
    how it's counted, given c >> s for this project's ~96-channel MUA
    data (c^3 alone is ~885,000 for c=96, vs. s^3 ~216 for s=6) --
    matching what this project already found empirically (KF's real
    measured latency/energy being dominated by exactly this per-timestep
    c x c inversion, not by the comparatively tiny s x s state algebra).

    Small elementwise matrix additions/subtractions (+W, +Q, I-KH, the
    residual subtraction) are counted separately as 'elementwise_ops',
    NOT folded into the MAC total or its energy -- they're genuinely
    negligible next to the cubic matmul/inversion terms here, and
    keeping them separate keeps the accounting auditable rather than
    silently padding the MAC count with a different kind of operation.
    """
    A, W, H, Q = model.model
    A = np.asarray(A)
    H = np.asarray(H)
    s = A.shape[0]       # state dimensionality
    c = H.shape[0]        # measurement/channel dimensionality

    # --- per-timestep MACs, one term per line of predict()'s own update ---
    macs_Pm = s**3 + s**3                    # A@P (s^3) then (A@P)@A.T (s^3)
    macs_state_m = s**2                      # A@state
    macs_HPm = c * s**2                      # H@P_m
    macs_HPmHT = c**2 * s                    # (H@P_m)@H.T
    macs_inversion = c**3                    # MAC-equivalent cost of inv(c x c) -- APPROXIMATION, see docstring
    macs_K1 = s**2 * c                       # P_m@H.T
    macs_K2 = s * c**2                       # (P_m@H.T)@inv_term
    macs_KH = s**2 * c                       # K@H
    macs_P_update = s**3                     # (I-KH)@P_m
    macs_H_state_m = c * s                   # H@state_m
    macs_K_resid = s * c                     # K@residual

    macs_per_step = (macs_Pm + macs_state_m + macs_HPm + macs_HPmHT + macs_inversion +
                      macs_K1 + macs_K2 + macs_KH + macs_P_update +
                      macs_H_state_m + macs_K_resid)

    elementwise_per_step = s**2 + c**2 + s**2 + c + s  # +W, +Q, I-KH, Z-Hstate_m, state_m+=

    total_macs = macs_per_step * n_timesteps
    total_elementwise = elementwise_per_step * n_timesteps

    return _finalize(total_macs, 0, elementwise_ops=total_elementwise, n_samples=n_timesteps)


# --------------------------------------------------------------------------
# WF -- a single dense linear projection (sklearn LinearRegression/Lasso/
# Ridge/ElasticNet), no nonlinearity. Sparsity-exploitable on the INPUT
# side (MUA spike-count windows are often genuinely sparse) -- every op
# is still a MAC (the input is a real-valued spike COUNT, not a binary
# spike, so a genuine multiply is needed even for a nonzero-but-small
# value).
# --------------------------------------------------------------------------

def estimate_ops_wf(model, X):
    """model: a fitted bmi.decoders.WienerDecoder (model.model is the
    underlying sklearn LinearRegression/Lasso/Ridge/ElasticNet).
    X: the ACTUAL (scaled, windowed-and-flattened) input array as fed to
    model.predict(X) in test_all_decoders.py's load_wf_decoder() -- shape
    (n_samples, n_features). Counts effective MACs from X's REAL nonzero
    pattern, not an assumed-dense worst case.
    """
    X = np.asarray(X)
    n_outputs = np.atleast_2d(model.model.coef_).shape[0]
    nnz_total = int(np.count_nonzero(X))
    total_macs = nnz_total * n_outputs  # each nonzero input feature contributes
    # n_outputs MACs (one per output this decoder produces), matching the same
    # "nonzero input -> fan_out MACs" rule used for MLP/LSTM/QRNN below
    return _finalize(total_macs, 0, n_samples=X.shape[0])


# --------------------------------------------------------------------------
# MLP -- Keras Sequential of Dense(+ReLU)/Dropout layers. Sparsity-
# exploitable at EVERY layer, not just the input: ReLU's own output is
# genuinely sparse (every negative pre-activation becomes exactly zero),
# so deeper layers can have exploitable zeros too, not only the first.
# --------------------------------------------------------------------------

def _run_layers_capturing(model, X, capture_types):
    """Runs X through model.layers ONE LAYER AT A TIME (not via Keras's
    Model(inputs=, outputs=) graph-introspection API, which turned out
    to be unreliable across Keras versions for a plain Sequential built
    via .build() rather than an explicit functional Input -- confirmed
    directly: even calling the model once first didn't populate
    model.input/layer.output the way older Keras versions did). Only
    depends on model.layers and calling each layer as a plain callable,
    both stable across Keras generations.

    Yields (layer, layer_input_array) for every layer whose
    __class__.__name__ is in capture_types, in order -- layer_input_array
    is exactly what that layer received, captured BEFORE calling it.
    Dropout layers are skipped structurally (Keras Dropout layers are
    no-ops when called without training=True, which is the default --
    matching real inference behavior, not an approximation).
    """
    import tensorflow as tf
    x = tf.convert_to_tensor(X)
    for layer in model.layers:
        if layer.__class__.__name__ in capture_types:
            yield layer, np.asarray(x)
        x = layer(x)


def _dense_layers_with_inputs(model, X):
    """Runs X through `model` layer by layer (see _run_layers_capturing()),
    yielding (layer, layer_input_array) for every Dense layer in order --
    the raw model input for the first Dense layer, the previous Dense
    layer's real (already-ReLU'd, so possibly sparse) output for every
    layer after that. Dropout layers are no-ops at inference and don't
    change what the next Dense layer actually receives.
    """
    yield from _run_layers_capturing(model, X, {'Dense'})


def estimate_ops_mlp(model, X):
    """model: a compiled+loaded Keras Sequential (bmi.decoders.MLPDecoder's
    return value). X: the ACTUAL scaled input array as fed to
    model.predict(X) in test_all_decoders.py's load_dl_decoder().
    """
    total_macs = 0
    n_samples = np.asarray(X).shape[0]
    for layer, layer_input in _dense_layers_with_inputs(model, X):
        out_features = layer.get_weights()[0].shape[1]
        nnz = int(np.count_nonzero(layer_input))
        total_macs += nnz * out_features
    return _finalize(total_macs, 0, n_samples=n_samples)


# --------------------------------------------------------------------------
# LSTM -- Keras Sequential of LSTM(+Dense output) layers. Each LSTM layer
# has 4 gates (input/forget/cell-candidate/output), each computing
# W_x @ x_t + W_h @ h_{t-1} + b. Sparsity-exploitable ONLY on the W_x @ x_t
# (input) term, and only for the FIRST LSTM layer (raw spike-count input,
# genuinely sparse) -- a later stacked LSTM layer's input is the PREVIOUS
# layer's hidden state, which is tanh/sigmoid-activated and continuous
# (not sparse) in general, so its own W_x @ x_t term is treated as fully
# dense too. The RECURRENT W_h @ h_{t-1} term is ALWAYS fully dense
# (hidden state is continuous), for every layer, every timestep.
# --------------------------------------------------------------------------

def estimate_ops_lstm(model, X):
    """model: a compiled+loaded Keras Sequential (bmi.decoders.LSTMDecoder's
    return value) -- one or more stacked LSTM layers, then a Dense output
    layer. X: the ACTUAL scaled, windowed 3D input array (n_samples,
    timesteps, channels) as fed to model.predict(X) in
    test_all_decoders.py's load_dl_decoder() (windowed via
    bmi.preprocessing.transform_data() for lstm/qrnn specifically).
    """
    X = np.asarray(X)
    n_samples, timesteps, _ = X.shape

    total_macs = 0
    for layer, layer_input in _run_layers_capturing(model, X, {'LSTM', 'Dense'}):
        if layer.__class__.__name__ == 'LSTM':
            kernel, recurrent_kernel, _bias = layer.get_weights()
            units = recurrent_kernel.shape[0]
            gate_width = kernel.shape[1]  # 4*units, but read directly rather than assumed

            # Sparsity-aware INPUT term: nnz per timestep x gate_width,
            # summed over every (sample, timestep) -- matches Dense/WF's
            # own "nonzero input -> fan_out MACs" rule, applied per
            # timestep here.
            nnz_per_step = np.count_nonzero(layer_input, axis=-1)  # (n_samples, timesteps)
            total_macs += int(nnz_per_step.sum()) * gate_width

            # ALWAYS-dense RECURRENT term: W_h @ h_{t-1}, every timestep,
            # every sample -- hidden state is continuous, no sparsity to
            # exploit.
            total_macs += n_samples * timesteps * units * gate_width
        else:  # Dense -- the output layer, input is the last LSTM layer's
            # (non-sparse) final hidden state, not raw spike data, so
            # treated as fully dense (still counted via real nnz for
            # consistency/auditability, it just won't be exploitably
            # sparse in practice)
            out_features = layer.get_weights()[0].shape[1]
            nnz = int(np.count_nonzero(layer_input))
            total_macs += nnz * out_features

    return _finalize(total_macs, 0, n_samples=n_samples)


# --------------------------------------------------------------------------
# QRNN -- bmi.decoders.QRNN's own custom layer (NOT Keras's built-in
# anything). preprocess_input() does one causal Conv2D producing 3*units
# channels (z/f/o gates) over the WHOLE padded sequence at once; step()
# is then a cheap, WEIGHT-FREE elementwise recurrence over those
# precomputed gate values (no matmul at all -- see decoders.py's QRNN.step()).
# So the entire per-layer weighted-sum cost is the conv; step()'s
# elementwise ops are reported separately, not as MAC/ACC.
# --------------------------------------------------------------------------

def estimate_ops_qrnn(model, X):
    """model: a compiled+loaded Keras Sequential (bmi.decoders.QRNNDecoder's
    return value) -- one or more stacked custom QRNN layers, then a Dense
    output layer. X: the ACTUAL scaled, windowed 3D input array (n_samples,
    timesteps, channels), same convention as estimate_ops_lstm().

    Conv MAC counting under CAUSAL left-padding (window_size-1 zeros
    prepended, matching QRNN.preprocess_input()'s own
    K.temporal_padding(inputs, (window_size-1, 0))): raw timestep i (0-
    indexed, T total timesteps) ends up inside min(T - i, window_size)
    output windows -- window_size for every INTERIOR position, but FEWER
    for the last (window_size - 1) positions, since padding is only on
    the LEFT and there's nothing on the right to keep a late position
    fully reused. (An earlier version of this function assumed a
    uniform window_size for every position -- caught by testing against
    a brute-force direct simulation, which disagreed at exactly these
    boundary positions; NOT a hypothetical concern, a real bug this
    function's own test suite catches.) Verified exactly against that
    brute-force simulation across multiple window sizes and sparsity
    levels (see this module's test).
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    total_macs = 0
    for layer, layer_input in _run_layers_capturing(model, X, {'QRNN', 'Dense'}):
        if layer.__class__.__name__ == 'QRNN':
            kernel = layer.get_weights()[0]  # (window_size, 1, input_dim, 3*units)
            window_size = kernel.shape[0]
            out_channels = kernel.shape[-1]  # 3*units

            T = layer_input.shape[1]
            nnz_per_timestep = np.count_nonzero(layer_input, axis=-1)  # (n_samples, T)
            multiplicities = np.minimum(T - np.arange(T), window_size)  # (T,) -- see docstring
            weighted_nnz = int((nnz_per_timestep * multiplicities[np.newaxis, :]).sum())
            total_macs += out_channels * weighted_nnz
        else:  # Dense -- the output layer
            out_features = layer.get_weights()[0].shape[1]
            nnz = int(np.count_nonzero(layer_input))
            total_macs += nnz * out_features

    return _finalize(total_macs, 0, n_samples=n_samples)



# --------------------------------------------------------------------------
# SNN -- operation counting for this decoder now happens INLINE, inside
# model_bmi.py's SNN_Speck.forward() itself (see its count_ops parameter),
# not via an external estimate_ops_*() here like every other decoder above.
#
# THIS WAS A DELIBERATE CHANGE, not the original design: an earlier
# revision of this module had a standalone estimate_ops_snn(model, X) that
# re-walked the model externally, timestep by timestep, the same way
# estimate_ops_lstm()/estimate_ops_qrnn() do. That turned out to be the
# wrong shape of function for this specific model -- confirmed directly
# against model_bmi.py's real SNN_Speck class, not assumed:
#   1. SNN_Speck.forward() takes the WHOLE (T, N, C) trial in ONE call and
#      loops over T internally; there is no external per-timestep call to
#      hook into the way run_torch_model()'s own loop provides for the
#      other decoders.
#   2. forward() does substantially more than the four-layer IAF stack:
#      it also runs a cascaded EMA (temporal_decay_stages, sharing one
#      learned decay constant) and a position-weighted decode_output()
#      step, BOTH of which contribute real additional operations that an
#      external re-walk of self.layers alone would have silently missed
#      entirely.
# Counting inline, inside forward() itself, guarantees the tally can never
# drift from what forward() actually computes -- the alternative would
# have meant re-deriving forward()'s own control flow a second time here,
# with no guarantee the two stay in sync as forward() evolves.
#
# What's left here is just the same generic "raw counts -> memory
# accesses -> energy range" conversion every other decoder's estimate
# already goes through -- SNN's counts just arrive pre-computed from
# model_bmi.py rather than being derived by a function in this file.
# --------------------------------------------------------------------------

def finalize_snn_ops(mac, acc, elementwise, n_samples):
    """Converts the RAW mac/acc/elementwise counts returned by
    SNN_Speck.forward(x, count_ops=True) (see model_bmi.py) into the same
    {effective_macs, effective_accs, memory_accesses, energy_*, per_sample}
    shape every other decoder's estimate_ops_*() returns in this module --
    a thin wrapper around _finalize() so the SNN's externally-supplied
    counts go through the identical MAC/ACC -> memory-access -> energy
    conversion as everything else, for a genuinely apples-to-apples
    comparison across all six decoders.

    n_samples should be T * N (timesteps x batch size actually run in that
    forward() call) -- matching every other estimator's own per-sample
    normalization convention (dividing by however many individual
    timestep-predictions were produced, not by trial count).
    """
    return _finalize(mac, acc, elementwise_ops=elementwise, n_samples=n_samples)
