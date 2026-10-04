"""
Train the feedforward SNN velocity decoder on one dataset directory.

--data-path is a directory with train/ and test/ subfolders of .pkl trials
(export_snn_pkl.py, combine_snn_dataset.py or build_pretraining_pool.py
output), laid out as .../{experiment}/{subject}/... . The experiment
("bmi" or "hkm") selects the model module (models/model_bmi.py or
models/model_hkm.py) and, with the subject, the velocity scaling from
velocity_scalers.json.

Every trial is an independent example: the model resets its state at the
start of each forward pass. The loss is the RMSE between predicted and
true velocity in physical units.

Synaptic time constant: --tau-syn is off by default (no synaptic-current
stage). With a value, every spiking layer low-pass filters its input
current with a time constant (in 4 ms timesteps) initialized to that value
and trained with the weights, like the readout's EMA decay. The trained
per-layer values are part of the model state dict.

Outputs in --checkpoint-dir, with exp_name = bmi_{neuron_type}_{reset_type}:
    best_model_weights.pth                   lowest test loss so far
    checkpoint_{exp_name}_epoch{N}.pth       every --checkpoint-interval epochs
    final_model_{exp_name}.pth               written when training ends
    training_curves_{exp_name}.png, summary_{exp_name}.txt

Resume an interrupted run with --resume CHECKPOINT; start a new run from
another run's weights (fresh optimizer and epoch count) with
--init-weights-from CHECKPOINT.
"""

import argparse
import importlib
import os
import sys
import time

import sinabs
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import create_dataloaders, experiment_and_subject_from_path, get_velocity_scalers
from utils import EarlyStopping, load_checkpoint, plot_training_curves, save_checkpoint

STEP_MS = 4.0  # native sampling interval


def str2bool(value):
    return str(value).lower() in ("1", "true", "yes", "y")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config; its values apply unless the flag is given explicitly")

    model = parser.add_argument_group("model")
    model.add_argument("--neuron-type", type=str, default="lif", choices=["iaf", "lif"])
    model.add_argument("--reset-type", type=str, default="hard", choices=["hard", "soft"])
    model.add_argument("--final-layer-reset-type", type=str, default=None, choices=["hard", "soft"],
                       help="Reset for the output layer (default: same as --reset-type)")
    model.add_argument("--tau-mem", type=float, default=12.0, help="LIF membrane time constant")
    model.add_argument("--tau-syn", type=float, default=None,
                       help="Initial synaptic time constant, trained per layer (default: none)")
    model.add_argument("--hidden-dims", type=int, nargs="+", default=None,
                       help="Hidden layer widths (default: 512 256 128)")
    model.add_argument("--spike-thresholds", type=float, nargs="+", default=None,
                       help="One threshold per spiking layer: len(hidden_dims) + 1")
    model.add_argument("--spike-fn", type=str, default="single", choices=["multi", "single"])
    model.add_argument("--min-vmem", type=int, default=None, help="Lower bound on membrane potential")
    model.add_argument("--use-iaf-squeeze", action="store_true",
                       help="Use sinabs' Squeeze neuron layers")
    model.add_argument("--weight-init", type=str, default=None, choices=["kaiming", "xavier"])
    model.add_argument("--surrogate-grad", type=str, default="periodic_exponential",
                       choices=["periodic_exponential", "single_exponential", "gaussian",
                                "multi_gaussian", "heaviside"])
    model.add_argument("--last-layer-reset", action="store_true",
                       help="Reset the output layer's membrane before each timestep")
    model.add_argument("--n-bins", type=int, default=18,
                       help="Population-code bins per velocity axis (output width = 2 x n_bins)")
    model.add_argument("--temporal-decay-init", type=float, default=0.8,
                       help="Initial decay of the readout's EMA smoothing")
    model.add_argument("--learnable-temporal-decay", type=str2bool, default=True)
    model.add_argument("--temporal-decay-stages", type=int, default=1,
                       help="Number of cascaded EMA stages in the readout")
    model.add_argument("--use-exodus", type=str2bool, default=None,
                       help="Use sinabs-exodus CUDA kernels (default: auto-detect)")
    model.add_argument("--use-spikingjelly", action="store_true",
                       help="Use SpikingJelly neurons instead of sinabs")

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", type=str, required=True,
                      help="Directory with train/ and test/ .pkl trials")
    data.add_argument("--experiment", type=str, default=None, choices=["bmi", "hkm"],
                      help="Default: derived from --data-path")
    data.add_argument("--batch-size", type=int, default=20)
    data.add_argument("--num-workers", type=int, default=20)
    data.add_argument("--train-data-min", type=float, default=None,
                      help="Use only the first N minutes of training trials")
    data.add_argument("--velocity-lo", type=float, default=None,
                      help="Velocity mapped to --velocity-margin (default: from velocity_scalers.json)")
    data.add_argument("--velocity-hi", type=float, default=None,
                      help="Velocity mapped to 1 - --velocity-margin (default: from velocity_scalers.json)")
    data.add_argument("--velocity-margin", type=float, default=None)

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.001)
    train.add_argument("--channel-lasso-weight", type=float, default=0.0,
                       help="Group-lasso penalty on the first layer's input channels (0 = off)")
    train.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    train.add_argument("--scheduler-patience", type=int, default=10,
                       help="Epochs without improvement before the LR is reduced")
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument("--lr-factor", type=float, default=0.1)
    train.add_argument("--max-lr-reductions", type=int, default=3,
                       help="With --disable-early-stopping: stop after this many LR reductions")
    train.add_argument("--disable-early-stopping", action="store_true")
    train.add_argument("--use-amp", action="store_true")
    train.add_argument("--loss-eps", type=float, default=1e-8,
                       help="Added inside sqrt(MSE) to keep the gradient finite at 0")

    ckpt = parser.add_argument_group("checkpoints")
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ckpt.add_argument("--checkpoint-interval", type=int, default=10)
    ckpt.add_argument("--resume", type=str, default=None,
                      help="Continue an interrupted run (restores optimizer and epoch)")
    ckpt.add_argument("--init-weights-from", type=str, default=None,
                      help="Load only model weights from this checkpoint (fine-tuning); the "
                           "architecture flags must match it")

    log = parser.add_argument_group("logging")
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="bmi-snn")
    log.add_argument("--wandb-entity", type=str, default=None)
    log.add_argument("--wandb-name", type=str, default=None)
    log.add_argument("--wandb-tags", nargs="+", default=[])
    log.add_argument("--wandb-mode", type=str, default="online",
                     choices=["online", "offline", "disabled"])
    log.add_argument("--device", type=str, default="cuda")
    log.add_argument("--seed", type=int, default=42)
    log.add_argument("--verbose", action="store_true")
    log.add_argument("--log-interval", type=int, default=50, help="Batches between log lines")
    return parser.parse_args()


def merge_config_args(args):
    """Apply --config values to args, except for flags given explicitly on
    the command line. Config sections are flattened: {training: {epochs: 5}}
    sets args.epochs; keys with no matching argument are ignored."""
    if not args.config:
        return args
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    explicit = {a[2:].split("=")[0].replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    items = []
    for key, value in config.items():
        items.extend(value.items() if isinstance(value, dict) else [(key, value)])
    for key, value in items:
        attr = key.replace("-", "_")
        if attr not in explicit and hasattr(args, attr):
            setattr(args, attr, value)
    return args


def unscale_velocity(v_scaled, lo, hi, margin):
    """Inverse of the dataloader's scaling v -> margin + (1 - 2 margin)(v - lo)/(hi - lo)."""
    return lo + (hi - lo) * (v_scaled - margin) / (1 - 2 * margin)


def run_epoch(model, loader, criterion, device, scale, loss_eps, optimizer=None, use_amp=False,
              channel_lasso_weight=0.0, log_interval=50, verbose=True, desc=""):
    """One pass over `loader`; trains if an optimizer is given, else evaluates.

    Returns (mean loss, metrics dict). Loss is RMSE in physical velocity
    units (plus the channel-lasso penalty when training with it)."""
    training = optimizer is not None
    model.train(training)
    grad_scaler = torch.cuda.amp.GradScaler() if (training and use_amp) else None
    first_linear = model.layers[0] if channel_lasso_weight > 0 else None
    sums = {k: 0.0 for k in ("loss", "pred_loss", "channel_penalty", "velocity_error",
                             "membrane_mean", "membrane_max", "non_zero_ratio", "spike_rate")}

    for batch_idx, (_, inputs, targets) in enumerate(tqdm(loader, desc=desc, disable=not verbose)):
        inputs = inputs.transpose(0, 1).to(device)    # [N, T, C] -> [T, N, C]
        targets = targets.transpose(0, 1).to(device)  # [N, T, 2] -> [T, N, 2]
        T, N = inputs.shape[:2]

        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            outputs, _, total_spikes = model(inputs)
            outputs_phys = unscale_velocity(outputs, *scale)
            targets_phys = unscale_velocity(targets, *scale)
            pred_loss = torch.sqrt(criterion(outputs_phys, targets_phys) + loss_eps)
            loss = pred_loss
            if training and channel_lasso_weight > 0:
                penalty = torch.norm(first_linear.weight, dim=0).sum()  # L2 per input channel
                loss = pred_loss + channel_lasso_weight * penalty
                sums["channel_penalty"] += penalty.item()

        if training:
            optimizer.zero_grad()
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.no_grad():
            spike_rate = (total_spikes / (T * N * model.total_neuron_units)).item()
            sums["loss"] += loss.item()
            sums["pred_loss"] += pred_loss.item()
            sums["velocity_error"] += torch.sqrt(
                ((outputs_phys - targets_phys) ** 2).sum(dim=-1)).mean().item()
            sums["membrane_mean"] += outputs.mean().item()
            sums["membrane_max"] += outputs.abs().max().item()
            sums["non_zero_ratio"] += (outputs != 0).float().mean().item()
            sums["spike_rate"] += spike_rate
        if training and (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] loss={loss.item():.5f} "
                  f"spike_rate={spike_rate:.4f}")

    metrics = {k: v / len(loader) for k, v in sums.items()}
    if training:
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics["loss"], metrics


def n_trials_for_minutes(dataset, minutes, first_trial_timesteps):
    """Number of leading training trials that make up `minutes` of data.

    Fixed-length trials (bmi) use the first trial's length; whole-trial
    datasets (hkm) have variable lengths, so their actual lengths are summed
    and the first trial reaching the target is included."""
    target_timesteps = minutes * 60000.0 / STEP_MS
    if dataset.experiment != "hkm":
        return int(round(target_timesteps / first_trial_timesteps))
    total = 0
    for i in range(len(dataset)):
        total += dataset.trial_length(i)
        if total >= target_timesteps:
            return i + 1
    return len(dataset)


def build_loaders(args, experiment, subject):
    """Train/test loaders. The test loader never drops trials; it batches
    whole test trials only when they all have the same length."""
    train_loader, test_loader = create_dataloaders(
        data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle_train=True, experiment=experiment, subject=subject)

    trial_timesteps = train_loader.dataset[0][1].shape[0]
    print(f"First train trial length: {trial_timesteps} timesteps ({trial_timesteps * STEP_MS:.0f} ms)")
    if args.train_data_min is not None:
        n_trials = n_trials_for_minutes(train_loader.dataset, args.train_data_min, trial_timesteps)
        subset = Subset(train_loader.dataset, range(min(n_trials, len(train_loader.dataset))))
        train_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False,
                                  num_workers=train_loader.num_workers,
                                  pin_memory=train_loader.pin_memory,
                                  collate_fn=train_loader.collate_fn)

    test_set = test_loader.dataset
    test_lengths = {test_set[i][1].shape[0] for i in range(len(test_set))}
    test_batch_size = min(args.batch_size, len(test_set)) if len(test_lengths) == 1 else 1
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, drop_last=False,
                             num_workers=test_loader.num_workers, pin_memory=test_loader.pin_memory,
                             collate_fn=test_loader.collate_fn)

    for name, loader in (("train", train_loader), ("test", test_loader)):
        if len(loader) == 0:
            raise ValueError(f"{name} loader has no batches ({len(loader.dataset)} trials, "
                             f"batch_size={loader.batch_size})")
    print(f"Dataset: {len(train_loader.dataset)} train trials, {len(test_set)} test trials "
          f"(test batch size {test_batch_size})")
    return train_loader, test_loader


def build_model(args, experiment, num_input_channels, device):
    model_module = importlib.import_module(f"models.model_{experiment}")
    model = model_module.create_model(
        use_spikingjelly=args.use_spikingjelly,
        last_layer_reset=args.last_layer_reset,
        spike_fn=sinabs.activation.MultiSpike if args.spike_fn == "multi" else sinabs.activation.SingleSpike,
        weight_init=args.weight_init,
        min_vmem=args.min_vmem,
        neuron_type=args.neuron_type,
        tau_mem=args.tau_mem,
        reset_type=args.reset_type,
        final_layer_reset_type=args.final_layer_reset_type,
        surrogate_grad=args.surrogate_grad,
        use_exodus=args.use_exodus,
        use_iaf_squeeze=args.use_iaf_squeeze,
        n_bins=args.n_bins,
        spike_thresholds=args.spike_thresholds,
        temporal_decay_init=args.temporal_decay_init,
        learnable_temporal_decay=args.learnable_temporal_decay,
        temporal_decay_stages=args.temporal_decay_stages,
        num_input_channels=num_input_channels,
        hidden_dims=args.hidden_dims,
        tau_syn=args.tau_syn,
        velocity_lo=args.velocity_lo,
        velocity_hi=args.velocity_hi,
        velocity_margin=args.velocity_margin,
    ).to(device)
    return model_module, model


def main():
    args = merge_config_args(parse_arguments())
    if args.resume and args.init_weights_from:
        raise ValueError("--resume and --init-weights-from are mutually exclusive")

    derived_experiment, subject = experiment_and_subject_from_path(args.data_path)
    experiment = args.experiment or derived_experiment
    if None in (args.velocity_lo, args.velocity_hi, args.velocity_margin):
        lo, hi, margin = get_velocity_scalers(experiment, subject)
        args.velocity_lo = lo if args.velocity_lo is None else args.velocity_lo
        args.velocity_hi = hi if args.velocity_hi is None else args.velocity_hi
        args.velocity_margin = margin if args.velocity_margin is None else args.velocity_margin

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    exp_name = f"bmi_{args.neuron_type}_{args.reset_type}"
    if args.neuron_type == "lif":
        exp_name += f"_tau{args.tau_mem}"

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_name or exp_name, config=vars(args), mode=args.wandb_mode,
                   tags=args.wandb_tags + [args.neuron_type, args.reset_type])

    print(f"Experiment: {exp_name} | data: {experiment}/{subject} | device: {device}")
    print(f"Hidden dims: {args.hidden_dims} | thresholds: {args.spike_thresholds} | "
          f"tau_syn: {args.tau_syn} | decay stages: {args.temporal_decay_stages}")
    print(f"Velocity scaling: lo={args.velocity_lo}, hi={args.velocity_hi}, "
          f"margin={args.velocity_margin}")
    print(f"Batch size: {args.batch_size} | lr: {args.lr} | epochs: {args.epochs}")

    train_loader, test_loader = build_loaders(args, experiment, subject)
    input_shape = test_loader.dataset[0][1].shape[1:]
    model_module, model = build_model(args, experiment, input_shape[0], device)

    if args.init_weights_from:
        source = torch.load(args.init_weights_from, map_location=device)
        model_module.load_model_weights(model, source["model_state_dict"],
                                        neuron_type=args.neuron_type,
                                        source_description=args.init_weights_from)
        print(f"Initialized weights from {args.init_weights_from}")
    if args.verbose:
        print(f"\nModel:\n{model}\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                  patience=args.scheduler_patience, min_lr=args.min_lr)
    early_stopping = None if args.disable_early_stopping else EarlyStopping(patience=args.patience)

    start_epoch, best_loss, lr_reduction_count = 0, float("inf"), 0
    train_losses, test_losses, lr_history = [], [], []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        train_losses = checkpoint.get("train_losses", [])
        test_losses = checkpoint.get("test_losses", [])
        lr_history = checkpoint.get("lr_history", [])
        lr_reduction_count = checkpoint.get("lr_reduction_count", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
        if start_epoch >= args.epochs:
            raise ValueError(f"Checkpoint is already at epoch {start_epoch} >= --epochs "
                             f"{args.epochs}; raise --epochs to continue training.")

    def checkpoint_info(**extra):
        return {"input_shape": input_shape, "best_loss": best_loss, "train_losses": train_losses,
                "test_losses": test_losses, "lr_history": lr_history,
                "lr_reduction_count": lr_reduction_count, "args": vars(args),
                "experiment": experiment, **extra}

    scale = (args.velocity_lo, args.velocity_hi, args.velocity_margin)
    training_start = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, scale, args.loss_eps, optimizer=optimizer,
            use_amp=args.use_amp, channel_lasso_weight=args.channel_lasso_weight,
            log_interval=args.log_interval, verbose=args.verbose, desc=f"Epoch {epoch + 1} [train]")
        test_loss, test_metrics = run_epoch(
            model, test_loader, criterion, device, scale, args.loss_eps,
            verbose=args.verbose, desc=f"Epoch {epoch + 1} [test]")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            lr_reduction_count += 1
            print(f"Learning rate reduced {old_lr:.6f} -> {new_lr:.6f} (#{lr_reduction_count})")
        lr_history.append(new_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | train {train_loss:.5f} | "
              f"test {test_loss:.5f} | velocity error {test_metrics['velocity_error']:.5f} | "
              f"spike rate {test_metrics['spike_rate']:.4f} | lr {train_metrics['learning_rate']:.6f}")
        if args.wandb:
            wandb.log({"epoch": epoch + 1, "epoch_time": epoch_time,
                       **{f"train/{k}": v for k, v in train_metrics.items()},
                       **{f"test/{k}": v for k, v in test_metrics.items()}}, step=epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            save_checkpoint(model, optimizer, epoch, test_loss,
                            os.path.join(args.checkpoint_dir, "best_model_weights.pth"),
                            additional_info=checkpoint_info(is_best=True))
            print(f"Saved best model with loss {best_loss:.5f}")
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, test_loss,
                            os.path.join(args.checkpoint_dir, f"checkpoint_{exp_name}_epoch{epoch + 1}.pth"),
                            additional_info=checkpoint_info())

        if args.disable_early_stopping:
            if new_lr <= args.min_lr and lr_reduction_count >= args.max_lr_reductions:
                print(f"Reached minimum LR after {lr_reduction_count} reductions; stopping.")
                break
        elif early_stopping(test_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - training_start
    print(f"\nTraining finished in {training_time / 60:.1f} min; best test loss {best_loss:.5f}")
    final_path = os.path.join(args.checkpoint_dir, f"final_model_{exp_name}.pth")
    save_checkpoint(model, optimizer, epoch, test_losses[-1] if test_losses else float("nan"),
                    final_path, additional_info=checkpoint_info(is_final=True))
    plot_training_curves(train_losses, test_losses,
                         os.path.join(args.checkpoint_dir, f"training_curves_{exp_name}.png"),
                         title=f"Training Progress - {exp_name}", lr_history=lr_history)
    with open(os.path.join(args.checkpoint_dir, f"summary_{exp_name}.txt"), "w") as f:
        f.write(f"Experiment: {exp_name}\nFinal epoch: {epoch + 1}\n"
                f"Best test loss: {best_loss:.5f}\nTraining time: {training_time / 60:.1f} minutes\n"
                f"Arguments: {vars(args)}\n")
    if args.wandb:
        wandb.run.summary.update({"best_test_loss": best_loss, "total_training_time": training_time})
        wandb.finish()
    print(f"Results saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
