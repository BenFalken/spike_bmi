"""
Training regimes the lab compares. Each is a dict of overrides on BASE (the
per-session Speck network of sbatch_scripts/snn_config.sh, trained by
train_snn.py); add your own here and name them with lab.py run --regimes.

Keys:
    spike_fn            'multi' (several spikes per neuron per timestep) or 'single'
    spike_thresholds    one per spiking layer (hidden layers, then output)
    hidden_dims         hidden layer widths
    reset_type          'hard' or 'soft'
    spike_rate_weight   train_snn.py --spike-rate-weight (0 = off)
    spike_rate_target   one rate (spikes per neuron per timestep) or one per spiking layer
    lr, weight_decay, epochs
    select              'pytorch' (default): keep the epoch with the best validation RMSE;
                        'emulated': the best validation RMSE on the emulated chip
    init                checkpoint to start from (fine-tuning: weights only)
"""

BASE = {
    'spike_fn': 'multi', 'spike_thresholds': [1.0, 1.0, 1.0, 1.0], 'hidden_dims': [256, 128, 64],
    'reset_type': 'hard', 'min_vmem': -1, 'n_bins': 18, 'temporal_decay_stages': 2,
    'spike_rate_weight': 0.0, 'spike_rate_target': [0.1], 'lr': 1e-3, 'weight_decay': 1e-3,
    'epochs': None, 'init': None, 'select': 'pytorch',
}

REGIMES = {
    # What the Jenkins per-session model was trained with: dense, ~3 output
    # spikes per neuron per timestep; chip predictions compressed to ~0.2.
    'baseline': {},
    # sbatch_scripts/snn_config.sh's hkm per_session setting.
    'rate10': {'spike_rate_weight': 10.0, 'spike_rate_target': [0.1, 0.1, 0.1, 0.25]},
    'rate30': {'spike_rate_weight': 30.0, 'spike_rate_target': [0.1, 0.1, 0.1, 0.25]},
    # Sparser targets, near the threshold-x2 condition the chip followed (slope 0.85).
    'rate10_sparse': {'spike_rate_weight': 10.0, 'spike_rate_target': [0.05, 0.05, 0.05, 0.1]},
    # At most one spike per neuron per timestep in training, as the chip fires
    # at most once per input event.
    'rate10_single': {'spike_fn': 'single', 'spike_rate_weight': 10.0,
                      'spike_rate_target': [0.1, 0.1, 0.1, 0.25]},
    # rate10, keeping the epoch whose emulated-chip validation RMSE is best.
    'rate10_emusel': {'spike_rate_weight': 10.0, 'spike_rate_target': [0.1, 0.1, 0.1, 0.25],
                      'select': 'emulated'},
}


def regime_config(name):
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; defined: {', '.join(REGIMES)}")
    cfg = dict(BASE)
    cfg.update(REGIMES[name])
    return cfg
