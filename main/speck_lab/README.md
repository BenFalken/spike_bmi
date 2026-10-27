# Speck lab

A self-contained sandbox for choosing the hkm SNN training regime before the real
hkm training runs. Nothing outside this folder uses it. It imports the model
(`models/model_hkm.py`) and the training loss pieces (`snn_training/train_snn.py`:
`spike_rate_penalty`, `unscale_velocity`) so a regime that wins here transfers
to `train_snn.py` unchanged.

**Why.** The Jenkins per-session model decodes at RMSE ~56 in PyTorch and ~118 on
Speck. `inference/speck_activity_study.py` traced that to network activity: the
chip updates each neuron per input event, and a network firing ~3 spikes per
output neuron per timestep gets its predictions compressed to ~1/5 on the chip.
The lab trains small models under different regimes and scores each one as

| version | what it is |
|---|---|
| `pytorch` | the trained network, one timestep per call |
| `quantized` | the network as deployed (integer weights and thresholds), one timestep per call |
| `emulated` | the quantized network run one event at a time, as the chip does (`emulator.py`) |
| `chip` | the real Speck2f (`--speck_devkit`) |

## Files

| file | |
|---|---|
| `emulator.py` | per-event emulation of the deployed network (integer weights from sinabs' own discretization) |
| `regimes.py` | the regimes compared; add your own here |
| `lab.py` | `validate`, `run`, `summary` |
| `plots.py` | `lab_rmse.png`, `lab_rates.png` |
| `run_lab.sbatch` | trains the regimes on the cluster, one array task each |

## Everything on the Speck laptop, in one go

```bash
cd main/speck_lab
DATA=~/path/to/data SESSION=sub-Jenkins_ses-20090912_behavior+ecephys bash run_lab_local.sh
```

`run_lab_local.sh` runs the steps below in sequence: validates the emulator against
the study's chip run, trains every regime one after another, scores each one as
PyTorch, quantized, emulated chip and on the chip, and writes the summary. It needs
the session's `train/` and `test/` splits under `$DATA/snn_datasets/hkm/<subject>/mua/`.
Options (environment): `REGIMES`, `N_TRAIN`, `N_VAL`, `N_TEST`, `EPOCHS`, `OUT_DIR`,
`DEVKIT` (`none` for host only), `SHUFFLE`, `SKIP_VALIDATE=1`; see the script's header.
Re-running resumes: trained regimes are only re-scored. The log is
`$OUT_DIR/run_lab_local.log`.

## 1. Validate the emulator (laptop, once)

Before trusting the emulator, compare it with the real chip run of the study, on the
same checkpoint and trials:

```bash
cd main/speck_lab
python lab.py validate \
  --checkpoint $D/snn_checkpoints/hkm/jenkins/per_session_binary/$S/best_model_weights.pth \
  --snn_dataset $D/snn_datasets/hkm/jenkins/mua/$S \
  --study_json ../../docs/data/speck_activity_jenkins_20090912.json
```

It prints, for the chip (from the study) and the emulator (with three event-order
choices), the output spike ratio over the quantized network (chip: 1.96), the
prediction slope (chip: 0.22) and the layer rates (chip: 0.131 0.630 1.273 6.065).
If one emulator row is close, use that `--emulator_shuffle` setting below. If none
is, the emulator misses something the chip does: score regimes on the chip only.

## 2. Train and compare regimes

On the cluster (no devkit needed; one task per regime):

```bash
cd main/speck_lab && mkdir -p lab_logs
sbatch --export=ALL,SNN_DATASET=$DATA_ROOT/snn_datasets/hkm/jenkins/mua/<session>,OUT_DIR=lab_runs/jenkins \
    --array=0-5 run_lab.sbatch
# afterwards
python lab.py summary --out_dir lab_runs/jenkins
```

or on one machine: `python lab.py run --snn_dataset ... --out_dir lab_runs/jenkins --regimes baseline rate10 rate30`.

Defaults: the first 80 training trials, the next 15 for picking the best epoch,
the first 30 test trials, 10 epochs. Then copy `lab_runs/jenkins` to the laptop
and add the real chip: the same `run` command with `--speck_devkit speck2fdevkit:0`
reuses the trained models and only scores them.

`--init <checkpoint>` fine-tunes every regime from an existing model instead of
training from scratch (e.g. the dense Jenkins per-session checkpoint).

## Regimes (`regimes.py`)

| regime | change from the deployed per-session config |
|---|---|
| `baseline` | none (MultiSpike, hard reset, thresholds 1, no penalty) |
| `rate10` | spike-rate penalty weight 10, targets 0.1 / 0.1 / 0.1 / 0.25 (what `snn_config.sh` now uses for hkm) |
| `rate30` | the same at weight 30 |
| `rate10_sparse` | weight 10, targets 0.05 / 0.05 / 0.05 / 0.1 |
| `rate10_single` | `rate10` with SingleSpike (at most one spike per neuron per timestep) |
| `rate10_emusel` | `rate10`, keeping the epoch with the best *emulated-chip* validation RMSE |
| `rate10_wd` | `rate10` with 10x the weight decay (1e-2): smaller weights, less cancellation |
| `rate10_cancel5`, `rate10_cancel20` | `rate10` plus a penalty (weight 5 or 20) on excitatory input cancelled by inhibitory input within a timestep |

### Why cancellation

Fine-tuning the dense Jenkins model with `rate10` cut its output rate 8x at almost no
PyTorch cost, but the emulated chip still lost ~50 RMSE, while models trained from scratch
at similar rates lost ~8. Rate is not what the chip gets wrong: the chip integrates a
timestep's events one at a time, so a neuron whose excitation and inhibition cancel within
the step fires if the excitation arrives first. The cancellation penalty
(`model.track_cancel`, `cancel_weight`) trains against exactly that.

### Diagnostics and calibrated readout

Every scoring also reports, per layer of the quantized network: the share of neuron-steps
that are order-sensitive (excitation alone reaches threshold, the step's net input does
not), the cancelled drive per neuron-step and the mean |weight| (both in thresholds).

`emulated_cal` / `chip_cal`: a linear readout fitted to the emulated chip's output spikes on
the training trials (ridge strength chosen on the validation trials; kept only if it beats
the model's own readout there) and applied to the test trials. Like the EMA and population
readout, it runs on the host, so it can be deployed as is. `--no_calibrate` skips it.

Re-running `run` on an existing `--out_dir` re-scores the saved models with all of this,
without retraining.

### Hidden-layer-1 thresholds

Hidden layer 1 takes ~32 mixed-sign input events per timestep and is where the chip's error
starts (its order sensitivity and cancelled drive are the highest of any layer, and no rate
regime changes them). Two ways to raise its threshold:

- `lab.py probe --checkpoint <trained> --snn_dataset <session> [--calibrate]`: the trained
  network with hidden layer 1's threshold (and lower bound) scaled by 1, 1.5, 2, 3
  (`--factors`, `--layers`), no retraining. Shows how far the chip gap closes and what the
  untrained change costs in PyTorch. Add `--speck_devkit speck2fdevkit:0` to score the
  same models on the chip (`chip`, `chip_cal`).
- Regimes `rate10_h1thr2`, `rate10_h1thr3`: train (or fine-tune with `INIT`) with that
  threshold, so the network can recover its accuracy. Watch hidden layer 1's mean |weight|
  in thresholds: if training scales the weights back up by the same factor, the gain is lost.

Fine-tuning keeps the regime's thresholds: a checkpoint stores its thresholds in its state
dict, which would otherwise overwrite them on loading.

## Reading the results

`results.json`, the printed table and `lab_rmse.png` give each regime's RMSE as
PyTorch, quantized, emulated and chip; the slope of the chip versions' predictions on
the quantized network's (1 = no compression); output rates; and synaptic events per
timestep (spikes x fan-out), a proxy for chip energy. `lab_rates.png` shows the rate
of every layer per regime.

A regime is ready for `train_snn.py` when its chip (or validated emulator) RMSE is
close to its PyTorch RMSE and its PyTorch RMSE is close to the baseline's. Put its
settings in `sbatch_scripts/snn_config.sh` (`PER_SESSION_RATE_ARGS` for the penalty).

These are small, short runs (part of one session): use them to rank regimes, not to
report final accuracy.
