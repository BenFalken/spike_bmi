# spike_bmi

Decoding hand velocity from intracortical spiking activity with classical, deep-learning
and spiking decoders, and running the spiking decoder on SynSense's Speck2f
neuromorphic chip.

Each recording session is decoded by six decoders, compared on accuracy (RMSE in mm/s,
and correlation), latency per sample, and energy per sample:

| decoder | what it is |
|---|---|
| `kf` | Kalman filter |
| `wf` | Wiener filter |
| `lstm`, `qrnn` | recurrent networks (TensorFlow) |
| `snn` | feedforward integrate-and-fire SNN (sinabs, PyTorch) with an EMA readout |
| `speck` | the same SNN, quantized and deployed to a Speck2f devkit |

## Layout

```
main/
  preprocessing_training/  raw data -> datasets; trains and evaluates KF, WF, LSTM, QRNN
  nwb_conversion/          datasets for trial-structured NWB sessions (experiment "hkm")
  bmi/                     shared decoding code: preprocessing, decoders, CV loop, metrics
  snn_training/            train_snn.py: trains the SNN
  models/                  SNN model definitions (model_bmi.py, model_hkm.py)
  utils/                   early stopping, checkpoints, training curves for train_snn.py
  inference/               evaluation of every decoder, the report, Speck deployment and diagnosis
  sbatch_scripts/          Slurm jobs for every stage (most also run locally with bash)
  configs/                 SNN training defaults
docs/                      figures referenced below
```

Every script opens with a docstring saying what it reads, what it writes and how to run
it. Data live under one root (`BMI_DATA_ROOT` / `DATA_ROOT`), laid out as
`datasets/`, `snn_datasets/`, `snn_checkpoints/` and `results/`.

## Workflow

1. **Datasets and classical/DL decoders.** `preprocessing_training/single_subject_pipeline.py`
   runs one session end to end: raw `.mat` → binned spikes and kinematics (4 ms bins) →
   ANN and SNN datasets → KF, WF, LSTM and QRNN, trained on all data and per training
   duration. On the cluster: `sbatch_scripts/run_bmi_subject_pipeline_array.sbatch`.
   Every stage skips outputs that already exist.
2. **SNN training.** `snn_training/train_snn.py`, driven by:
   - `run_snn_sweep.sbatch`: an architecture/hyperparameter sweep, per session
     (`--plan`, then submit the printed command).
   - `run_snn_pooled_pretrain.sbatch`, then `run_snn_pooled_finetune_array.sbatch`:
     pretrain the "medium" network (256 → 128) on all sessions of a subject, then
     fine-tune it per session. `RESET_TYPE=hard|soft` and `TAU_SYN` select the variant
     (`snn_medium_config.sh`).
3. **Inference and report.** `bash sbatch_scripts/run_inference.sbatch --submit bmi indy`
   evaluates every decoder on every session (`inference/test_all_decoders.py`), then builds
   `combined_metrics*.json` and the efficiency, energy and 4x2 comparison figures
   (`inference/make_report.py`).
4. **Speck.** On the devkit-connected laptop, `bash sbatch_scripts/run_inference.sbatch --local bmi indy`
   runs the same evaluation in series with `speck` added. Copy the cluster's
   `sessions/*.json` in first to extend them. `inference/export_test_split.py` writes the
   test-only data the laptop needs.
5. **Speck diagnosis** (no devkit needed, from `main/inference`):
   - `diagnose_speck.py` scores each session's checkpoint as trained (`pytorch`) and as
     quantized for the chip, next to the saved `snn` and `speck` results. It also compares
     the chip's output spikes with the quantized network's step by step: spike ratio,
     agreement, delay, re-decoding and a cross-validated readout re-fit. It writes
     `speck_diagnosis.json`.
   - `probe_speck.py` (with the devkit) measures how a single chip neuron integrates and
     fires.
   - `compare_speck_runs.py` draws the figure below from up to three
     `speck_diagnosis.json` files.

## Findings

### Decoders (indy, 36 sessions, fine-tuned SNN with hard reset, Speck-connected laptop)

| decoder | RMSE | latency / sample | parameters | energy / sample |
|---|---|---|---|---|
| KF | 60.52 | 0.010 ms | 9,864 | 434 µJ (RAPL) |
| WF | 48.03 | 0.050 ms | 2,882 | 1,872 µJ (RAPL) |
| LSTM | 39.48 | 34.8 ms | 238,002 | 1.19 J (RAPL) |
| QRNN | **37.62** | 35.2 ms | 232,402 | 1.20 J (RAPL) |
| SNN (PyTorch, CPU) | 39.96 | 0.41 ms | 61,959 | 14,826 µJ (RAPL) |
| SNN on Speck2f | 45.36 | 1.27 ms | 61,952 | **2.66 µJ** (chip power monitor) |

- **Accuracy.** In PyTorch the SNN is as accurate as the LSTM and within 2.5 RMSE of the QRNN.
- **Energy.** On Speck it uses about 5,000× less energy per sample than the SNN on the laptop
  CPU, and about 450,000× less than the recurrent networks.
- **Cost.** About 5 RMSE of accuracy.
- **Measurement caveat.** RAPL measures the whole CPU package, while the chip's power monitor
  measures only the chip, so the energy ratios compare deployments, not arithmetic.

### Where Speck loses accuracy

Three SNN training runs were deployed to the same chip, all with IAF neurons. Neither
LIF neurons nor a synaptic stage (τ_syn) can be deployed to it.

![Speck comparison of three SNN training runs](docs/speck_run_comparison.png)

| run | PyTorch | quantized | Speck | chip adds | chip / quantized output spikes |
|---|---|---|---|---|---|
| per-session, hard reset | 43.36 | 43.89 | **45.78** | 1.9 | 1.9× |
| pretrained + fine-tuned, hard reset | 40.54 | 41.59 | **46.01** | 4.4 | 1.6× |
| pretrained + fine-tuned, soft reset | 38.80 | 39.82 | **47.99** | 8.2 | 1.3× |

(Means over sessions with chip results: 36, 36 and 35.)

**What was ruled out**
- **Quantization:** about 1 RMSE (8-bit weights, integer thresholds).
- **Decoding:** the chip's output spikes, re-decoded on the host, reproduce the Speck RMSE
  exactly.
- **Readout delay:** no delay makes the chip's spikes line up with the quantized network's.
  Correlation per step is about 0.03 with no shift and at most about 0.1 at any delay up to
  50 steps. Removing the best delay gains about 1 RMSE.
- **Readout calibration:** a readout re-fitted to the chip's spikes (cross-validated) does no
  better than about 47 in any run. The chip's output carries less information than the
  quantized network's, and no readout recovers it.
- **Soft reset:** on the chip it is worse than hard reset, although it is better in PyTorch.
- **Binarized input:** per-session models retrained with each input channel clipped to at
  most one spike per 4 ms bin reach Speck 45.67 on 30 sessions, with the same chip penalty
  (about 2 RMSE) and spike ratio (about 1.9×). Few bins held more than one spike, so the
  inputs barely changed, and capping events per channel leaves each neuron summing events from
  many channels per timestep.

**How the chip differs from training** (`probe_speck.py`, single neuron on a Speck2f devkit)
- **Training model:** each timestep's input is summed, then the neuron fires ⌊v/θ⌋ spikes
  and resets.
- **Chip, per event:** the neuron is updated after every input event and fires at most once
  per event.
- **Chip, reset:** the configured reset (hard or soft) is applied after each spike.
- **Chip, after a reset:** the first input event after a membrane reset is lost.

So within a timestep the chip's result depends on the order in which excitatory and
inhibitory events arrive. It also cannot emit several spikes for one large input. The chip
does keep membrane potential across timesteps: IAF, no leak, in integers.

**Interpretation.** Per-feature firing rates agree well (correlation 0.78–1.0), and so does
the EMA-smoothed output (0.8–0.96). The timing of individual spikes does not: per step the
chip and the quantized network agree at about 0.03.

The chip adds more error the more a model relies on stored state:

| run | stored state | chip adds |
|---|---|---|
| per-session | each spike wipes the membrane | 1.9 |
| fine-tuned | fewer spikes, longer-lived sub-threshold charge | 4.4 |
| soft reset | every spike's remainder is kept | 8.2 |

A model that carries charge forward also carries the chip's within-timestep errors forward,
and they compound across layers. The training changes that improve the PyTorch decoder act
through exactly the per-timestep dynamics the chip does not reproduce, so they do not survive
deployment. Speck lands at 45.8–48 whichever model is deployed.

**Conclusion.** The per-session, hard-reset checkpoints are the best case for Speck.
- **Speck:** 45.8 RMSE.
- **Same network in PyTorch:** 43.4, so within 2 RMSE.
- **Best PyTorch SNN:** the fine-tuned model at 40.5, about 5 better.

The remaining gap comes from the chip updating per event while training sums each timestep.
It does not come from quantization, decoding, delay or the readout. Two directions could close it and were not pursued:
- **Time bins short enough** that each neuron receives about one event per timestep, so that
  per-timestep training matches per-event updates.
- **Training through an event-by-event model** of the chip's neurons.

## Requirements

Python 3.11 with numpy, scipy, scikit-learn, h5py, matplotlib, Pillow, PyTorch, sinabs
(≥ 3) and TensorFlow (LSTM/QRNN). The Speck decoder also needs samna and a Speck2f devkit.
`diagnose_speck.py` and `compare_speck_runs.py` need sinabs but no devkit.
