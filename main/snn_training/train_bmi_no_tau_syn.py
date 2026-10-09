"""
Training script for the BMI feedforward SNN velocity decoder.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import os
import sys
import time

import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
import sinabs
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_bmi import create_model
from models.model_bmi import load_model_weights
from dataset import create_dataloaders, experiment_and_subject_from_path, get_velocity_scalers
from utils import EarlyStopping, save_checkpoint, load_checkpoint, plot_training_curves


def minutes_to_samples(minutes, step_ms):
    """Convert a duration in minutes to a number of trial samples, given
    the step size in ms.
    """
    return int(round(minutes * 60000.0 / step_ms))


def unscale_velocity(v_scaled, lo, hi, margin):
    """Inverse of the dataloader's forward scaling
    (margin + (1-2*margin) * (v - lo) / (hi - lo))
    """
    return lo + (hi - lo) * (v_scaled - margin) / (1 - 2 * margin)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the BMI feedforward SNN velocity decoder")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (overrides other arguments)")

    # Model arguments
    parser.add_argument("--use-iaf-squeeze", action="store_true", help="Use IAF squeeze layer")
    parser.add_argument("--neuron-type", type=str, default="lif", choices=["iaf", "lif"])
    parser.add_argument("--reset-type", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--final-layer-reset-type", type=str, default=None, choices=["hard", "soft"],
                        help="Reset mechanism for final layer neurons (default: same as other layers)")
    parser.add_argument("--tau-mem", type=float, default=12.0, help="Membrane time constant for LIF neurons")
    parser.add_argument("--use-exodus", type=bool, default=None,
                        help="Use sinabs-exodus for CUDA acceleration (default: auto-detect)")
    parser.add_argument("--use-spikingjelly", action="store_true", help="Use SpikingJelly instead of sinabs")
    parser.add_argument("--weight-init", type=str, default=None, choices=["kaiming", "xavier"])
    parser.add_argument("--spike-fn", type=str, default="single", choices=["multi", "single"])
    parser.add_argument("--min-vmem", type=int, default=None, help="Lower bound for membrane potential")
    parser.add_argument("--surrogate-grad", type=str, default="periodic_exponential",
                        choices=["periodic_exponential", "single_exponential", "gaussian",
                                 "multi_gaussian", "heaviside"])
    parser.add_argument("--last-layer-reset", action="store_true",
                        help="Reset vmem of last layer before each frame pass")
    parser.add_argument("--n-bins", type=int, default=18, help="Population-vector bins per axis (output width = 2x this)")
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=None,
                        help="Hidden layer widths, e.g. --hidden-dims 256 512 128 64 for a "
                             "5-layer network (default: None -> [512, 256, 128], the original "
                             "fixed 4-layer architecture).")
    parser.add_argument("--spike-thresholds", type=float, nargs='+', default=None,
                        help="One spike threshold per layer. Length must match "
                             "len(hidden_dims)+1")
    parser.add_argument("--temporal-decay-init", type=float, default=0.8)
    parser.add_argument("--learnable-temporal-decay", type=bool, default=True)
    parser.add_argument("--temporal-decay-stages", type=int, default=1,
                        help="Number of cascaded EMA stages feeding the readout (all sharing "
                             "the same learnable decay parameter). Default 1 reproduces the "
                             "original single-stage EMA exactly. >1 chains multiple EMAs,"
                             "a standard technique for building a gamma/Erlang-shaped kernel.")
    parser.add_argument("--tau-syn", type=float, default=None,
                        help="Synaptic current time constant (IAF and LIF both support this "
                             "-- confirmed directly via inspect.signature, not LIF-only as one "
                             "might assume). Adds a second filtering stage before the membrane "
                             "potential itself: input -> synaptic current (decaying at "
                             "tau_syn) -> membrane potential. Default None matches sinabs' own "
                             "default exactly (confirmed byte-for-byte backward compatible "
                             "when left unset).")

    # Velocity scaling (must match whatever CustomDataset.__getitem__
    # actually applies -- see unscale_velocity() above). Default None on
    # all three -- auto-derived from --data-path via dataset.py's own
    # get_velocity_scalers() (see main()) rather than hardcoded to one
    # subject's values, so this no longer needs to be kept in sync by
    # hand with whatever dataset.py's CustomDataset will actually apply.
    # Pass any of the three explicitly to override the auto-derived
    # value (e.g. for a one-off experiment); the other two still
    # auto-derive independently.
    parser.add_argument("--velocity-lo", type=float, default=None,
                        help="Physical velocity value the dataloader maps to --velocity-margin "
                             "(default: auto-derived from --data-path)")
    parser.add_argument("--velocity-hi", type=float, default=None,
                        help="Physical velocity value the dataloader maps to 1-velocity-margin "
                             "(default: auto-derived from --data-path)")
    parser.add_argument("--velocity-margin", type=float, default=None,
                        help="Scaled-space margin (dataloader maps [lo,hi] to [margin, 1-margin], "
                             "not literal [0,1]) (default: auto-derived from --data-path)")

    # Data arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to one session's exported .pkl directory, "
                             "e.g. .../datasets/bmi/mua/indy_20160407_02")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Independent trials per batch (every trial resets, no cross-"
                             "trial continuity to preserve).")
    parser.add_argument("--training-mode", type=str, default="windowed",
                         choices=["windowed"],
                        help="Kept as a CLI argument only so existing scripts/sbatch files "
                             "that already pass --training-mode windowed keep working "
                             "unchanged. There is only one mode now (every trial "
                             "independently resets state) -- 'continuous' and 'chunked' were "
                             "both tried and removed after real comparative results showed "
                             "windowed outperforming both; see module docstring. Passing "
                             "anything other than 'windowed' is an immediate argparse error, "
                             "not a silent fallback.")
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--train-data-min", type=int, default=None,
                        help="How much of the training data to use, in minutes")
    parser.add_argument("--train_on_small_dataset", action="store_true", help="Use small dataset (0.2x)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--channel-lasso-weight", type=float, default=0.0,
                        help="Group lasso penalty on the FIRST layer's input channels "
                             "(default 0.0, off). Penalizes sum_c ||W_first[:, c]||_2 -- each "
                             "channel's TOTAL combined weight across every postsynaptic unit "
                             "in the first hidden layer, as one group -- unlike --weight-decay "
                             "(L2 on every individual weight independently), this creates real "
                             "pressure to drive whole uninformative channels toward zero "
                             "together, not just shrink everything uniformly. See "
                             "check_channel_importance.py for the diagnostic that motivated "
                             "this: real trained checkpoints showed almost no channel "
                             "differentiation under weight decay alone.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--max-lr-reductions", type=int, default=3)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--loss-eps", type=float, default=1e-8,
                        help="Epsilon added before sqrt(MSE) to avoid an unbounded gradient near 0")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-weights-from", type=str, default=None,
                        help="Fine-tuning, NOT resume -- a genuinely different mechanism, "
                             "not --resume with a different name. Loads ONLY the model's "
                             "weights from this checkpoint path; optimizer state, epoch "
                             "count, best_loss, and LR-scheduler/early-stopping state all "
                             "start completely fresh, as a normal new training run would. "
                             "This matters concretely: --resume restores optimizer momentum "
                             "and CARRIES OVER best_loss from the source checkpoint -- for "
                             "fine-tuning on a DIFFERENT session's data than what the "
                             "checkpoint was trained on, stale momentum from unrelated "
                             "gradients is the wrong starting point, and a carried-over "
                             "best_loss from a different data distribution could mean this "
                             "run never saves a 'best' checkpoint at all if this session's "
                             "own loss never beats that stale number. The source "
                             "checkpoint's architecture (hidden_dims, neuron_type, "
                             "thresholds, etc.) MUST exactly match what THIS run's own CLI "
                             "args specify -- the model is constructed fresh from THIS run's "
                             "args, then weights are loaded on top; a mismatch fails loudly "
                             "via shape errors rather than silently loading wrong weights, "
                             "which is the safe behavior. Mutually exclusive with --resume.")

    # WandB arguments
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="bmi-snn")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=[])
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Other arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    return parser.parse_args()


def _explicitly_passed_flags():
    """Used by merge_config_args() so an explicit CLI
    override always wins over a config file's value for the same
    setting, rather than being silently discarded."""
    flags = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            name = arg[2:].split("=")[0]
            flags.add(name.replace("-", "_"))
    return flags


def merge_config_args(args):
    """Merge config file with command line arguments.

    CLI flags EXPLICITLY passed on the command line always win over the
    config file's value for the same setting -- previously the config
    file unconditionally overwrote every matching argument regardless of
    whether it was explicitly set on the CLI or left at its default,
    which silently discarded --epochs/--weight-decay/--batch-size/
    --neuron-type/--reset-type overrides (confirmed: a sweep script
    passing --epochs 500 --weight-decay 0.0 together with --config
    ended up running at the config's epochs=4000, weight_decay=0.001
    regardless -- visible directly in "Epoch 7/4000" appearing in output
    despite --epochs 500 being passed).
    """
    if args.config:
        config = load_config(args.config)
        explicit = _explicitly_passed_flags()

        for category, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    py_attr = key.replace("-", "_")
                    if py_attr in explicit:
                        continue  # CLI wins -- don't let config clobber an explicit override
                    if hasattr(args, py_attr):
                        setattr(args, py_attr, value)
            else:
                py_attr = category.replace("-", "_")
                if py_attr in explicit:
                    continue
                if hasattr(args, py_attr):
                    setattr(args, py_attr, values)
    return args


def train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                 use_amp=False, log_interval=50, use_wandb=False, verbose=True,
                 v_lo=-280.56, v_hi=316.54, v_margin=0.05, loss_eps=1e-8,
                 channel_lasso_weight=0.0):
    """Train for one epoch.

    channel_lasso_weight (default 0.0, off): adds
    channel_lasso_weight * sum_c ||W_first[:, c]||_2 to the loss --
    W_first is the FIRST layer's weight matrix (model.layers[0], always
    the input-facing Linear regardless of hidden_dims), and the sum is
    over its INPUT dimension (one term per input channel, each the
    combined L2 norm of that channel's connections to every postsynaptic
    unit in the first hidden layer). This is GENUINELY DIFFERENT from
    --weight-decay (L2 on every individual weight independently,
    uniformly across the whole network) -- group lasso penalizes each
    channel's TOTAL combined influence as one group, which is what
    actually creates pressure to drive whole channels toward zero
    together, rather than shrinking every weight a little regardless of
    whether it's part of an informative channel or a noisy one. Motivated
    directly by check_channel_importance.py finding real, trained
    checkpoints show almost no separation between channels (CV~0.05,
    essentially uniform) under plain weight decay alone.
    """
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    membrane_means, membrane_maxs, non_zero_ratios, spike_rates = [], [], [], []
    pred_losses, channel_penalties = [], []
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=not verbose)

    first_linear = model.layers[0] if channel_lasso_weight > 0 else None

    for batch_idx, (labels, inputs, targets) in enumerate(pbar):
        inputs = inputs.transpose(0, 1).to(device)   # [N,T,C] -> [T,N,C]
        targets = targets.transpose(0, 1).to(device)  # [N,T,2] -> [T,N,2]
        T, N = inputs.shape[0], inputs.shape[1]

        # Every trial (whatever its length -- 256ms or a concatenated
        # multiple thereof) is treated as an independent, self-contained
        # example, matching how the ANN decoders are trained -- this also
        # removes the batch_size=1/shuffle_train=False requirement that
        # continuity depended on -- real batching and shuffling are both
        # meaningful again now. model.forward() always resets state
        # internally (see model_bmi.py); no reset_state argument needed
        # here anymore.
        optimizer.zero_grad()

        def compute_loss(outputs, final_layer_spikes, total_spikes):
            targets_phys = unscale_velocity(targets, v_lo, v_hi, v_margin)
            outputs_phys = unscale_velocity(outputs, v_lo, v_hi, v_margin)
            pred_loss = torch.sqrt(criterion(outputs_phys, targets_phys) + loss_eps)
            spike_rate = total_spikes / (T * N * model.total_neuron_units)
            return pred_loss, spike_rate

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs, final_layer_spikes, total_spikes = model(inputs)
                pred_loss, spike_rate = compute_loss(outputs, final_layer_spikes, total_spikes)
        else:
            outputs, final_layer_spikes, total_spikes = model(inputs)
            pred_loss, spike_rate = compute_loss(outputs, final_layer_spikes, total_spikes)

        if channel_lasso_weight > 0:
            channel_penalty = torch.norm(first_linear.weight, dim=0).sum()
            total_loss = pred_loss + channel_lasso_weight * channel_penalty
            channel_penalties.append(channel_penalty.item())
        else:
            total_loss = pred_loss
        pred_losses.append(pred_loss.item())

        with torch.no_grad():
            membrane_mean = outputs.mean().item()
            membrane_max = outputs.abs().max().item()
            non_zero_ratio = (outputs != 0).float().mean().item()
        membrane_means.append(membrane_mean)
        membrane_maxs.append(membrane_max)
        non_zero_ratios.append(non_zero_ratio)
        spike_rates.append(spike_rate.item())

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss += total_loss.item()

        if (batch_idx + 1) % log_interval == 0:
            penalty_str = f"\tChannel Penalty: {channel_penalties[-1]:.5f}" if channel_lasso_weight > 0 else ""
            print(f"Train Epoch: {epoch} [{batch_idx+1}/{total_batches} "
                  f"({100. * (batch_idx+1)/total_batches:.0f}%)]\t"
                  f"Task Loss: {total_loss.item():.5f}\tSpike Rate: {spike_rate.item():.4f}"
                  f"{penalty_str}")

    avg_loss = train_loss / total_batches
    avg_metrics = {
        "loss": avg_loss,
        "pred_loss": sum(pred_losses) / len(pred_losses),
        "membrane_mean": sum(membrane_means) / len(membrane_means),
        "membrane_max": sum(membrane_maxs) / len(membrane_maxs),
        "non_zero_ratio": sum(non_zero_ratios) / len(non_zero_ratios),
        "spike_rate": sum(spike_rates) / len(spike_rates),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    if channel_lasso_weight > 0:
        avg_metrics["channel_penalty"] = sum(channel_penalties) / len(channel_penalties)
    return avg_loss, avg_metrics

def evaluate_epoch(model, test_loader, criterion, device, epoch,
                    use_wandb=False, verbose=True,
                    v_lo=-280.56, v_hi=316.54, v_margin=0.05, loss_eps=1e-8):
    """Evaluate for one epoch. Every trial independently resets state
    unconditionally, matching train_epoch()'s windowed-only training --
    there is only one mode now, so no dispatch is needed here."""
    model.eval()
    test_loss = 0
    total_velocity_error = 0
    total_batches = len(test_loader)

    membrane_means, membrane_maxs, non_zero_ratios, spike_rates = [], [], [], []
    pbar = tqdm(test_loader, desc=f"Epoch {epoch+1} [Val]", disable=not verbose)

    with torch.no_grad():
        for batch_idx, (labels, inputs, targets) in enumerate(pbar):
            inputs = inputs.transpose(0, 1).to(device)
            targets = targets.transpose(0, 1).to(device)
            T, N = inputs.shape[0], inputs.shape[1]

            # Every trial is an independent example; there's no cross-
            # trial continuity to preserve or reason about. model.forward()
            # always resets state internally (see model_bmi.py); no
            # reset_state argument needed here anymore.
            outputs, _, total_spikes = model(inputs)
            spike_rate = (total_spikes / (T * N * model.total_neuron_units)).item()

            targets_phys = unscale_velocity(targets, v_lo, v_hi, v_margin)
            outputs_phys = unscale_velocity(outputs, v_lo, v_hi, v_margin)
            loss = torch.sqrt(criterion(outputs_phys, targets_phys) + loss_eps)

            membrane_mean = outputs.mean().item()
            membrane_max = outputs.abs().max().item()
            non_zero_ratio = (outputs != 0).float().mean().item()

            # Velocity error (L2 distance, in physical units) -- renamed
            # from "position_error": the quantity is hand VELOCITY, not
            # position.
            velocity_error = torch.sqrt(((outputs_phys - targets_phys) ** 2).sum(dim=-1)).mean().item()

            test_loss += loss.item()
            total_velocity_error += velocity_error
            membrane_means.append(membrane_mean)
            membrane_maxs.append(membrane_max)
            non_zero_ratios.append(non_zero_ratio)
            spike_rates.append(spike_rate)

            pbar.set_postfix({
                "loss": f"{loss.item():.5f}",
                "vel_err": f"{velocity_error:.3f}",
                "v_mean": f"{membrane_mean:.3f}",
            })

    avg_loss = test_loss / total_batches
    avg_velocity_error = total_velocity_error / total_batches
    avg_metrics = {
        "loss": avg_loss,
        "velocity_error": avg_velocity_error,
        "membrane_mean": sum(membrane_means) / len(membrane_means),
        "membrane_max": sum(membrane_maxs) / len(membrane_maxs),
        "non_zero_ratio": sum(non_zero_ratios) / len(non_zero_ratios),
        "spike_rate": sum(spike_rates) / len(spike_rates),
    }
    return avg_loss, avg_metrics


def setup_experiment(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    exp_name = f"bmi_{args.neuron_type}_{args.reset_type}"
    if args.neuron_type == "lif":
        exp_name += f"_tau{args.tau_mem}"

    if args.wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name or exp_name, config=vars(args),
            tags=args.wandb_tags + [args.neuron_type, args.reset_type],
            mode=args.wandb_mode,
        )
        wandb.config.update({
            "experiment_name": exp_name, "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        })

    return device, exp_name


def main():
    args = parse_arguments()
    args = merge_config_args(args)

    # Auto-derive velocity scaling from --data-path if not explicitly
    # set (via CLI or config) -- see get_velocity_scalers()'s own
    # docstring in dataset.py. Derives (experiment, subject) ONCE here
    # and reuses it for the create_dataloaders() call below, rather than
    # letting CustomDataset re-derive it a second, redundant time --
    # this also guarantees the dataloader's own scaling and the model's
    # velocity_lo/hi/margin are built from the EXACT SAME (experiment,
    # subject) pair, not two independently-derived ones that could
    # theoretically disagree if the derivation logic ever changes.
    velocity_experiment = velocity_subject = None
    if args.velocity_lo is None or args.velocity_hi is None or args.velocity_margin is None:
        velocity_experiment, velocity_subject = experiment_and_subject_from_path(args.data_path)
        auto_lo, auto_hi, auto_margin = get_velocity_scalers(velocity_experiment, velocity_subject)
        if args.velocity_lo is None:
            args.velocity_lo = auto_lo
        if args.velocity_hi is None:
            args.velocity_hi = auto_hi
        if args.velocity_margin is None:
            args.velocity_margin = auto_margin
        print(f"Auto-derived velocity scaling from --data-path (experiment={velocity_experiment}, "
              f"subject={velocity_subject}): lo={args.velocity_lo}, hi={args.velocity_hi}, "
              f"margin={args.velocity_margin} -- pass --velocity-lo/--velocity-hi/--velocity-margin "
              f"explicitly to override.")

    device, exp_name = setup_experiment(args)
    print(f"Starting experiment: {exp_name}")
    print(f"Sinabs or Spiking Jelly?: {'Sinabs' if not args.use_spikingjelly else 'Spiking Jelly'}")
    print(f"Device: {device}")
    print(f"Neuron type: {args.neuron_type}")
    print(f"Reset type: {args.reset_type}")
    print(f"Velocity scaling: lo={args.velocity_lo}, hi={args.velocity_hi}, margin={args.velocity_margin}")
    print(f"Training mode: {args.training_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total training data duration: {args.train_data_min} min")

    train_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        small=args.train_on_small_dataset,
        experiment=velocity_experiment,
        subject=velocity_subject,
    )

    # Input shape derived from test_loader (channel count is identical
    # either way). Trial DURATION for --train-data-min, however, is
    # derived from TRAIN_loader specifically, not test -- confirmed
    # directly this matters, not a style choice: for the grouped
    # datasets (mua_8_group_uniform etc.), train trials are short,
    # uniform group_size-chunked windows (e.g. 520 timesteps at
    # group_size=8), while test is a single, much longer whole
    # continuous stream (tens of thousands of timesteps) -- using the
    # TEST trial's duration here would silently select ~29x too few
    # train samples for --train-data-min 1 on this dataset (1 sample
    # instead of the correct ~29), a real, confirmed miscalibration this
    # wasn't already causing to be an issue for mua_large specifically,
    # only because THAT dataset's train and test trials happen to share
    # one uniform length -- moved up from later in this function
    # specifically so minutes_to_samples() below can use the REAL train
    # trial duration rather than a hardcoded guess.
    input_shape = test_loader.dataset[0][1].shape[1:]
    num_input_channels = input_shape[0]
    trial_n_timesteps = train_loader.dataset[0][1].shape[0]
    trial_duration_ms = trial_n_timesteps * 4.0  # this project's fixed native 4ms sampling
    print(f"Input shape: {input_shape} ({num_input_channels} channels)")
    print(f"Train trial duration (from actual data): {trial_n_timesteps} timesteps = {trial_duration_ms:.0f}ms")

    if args.train_data_min is not None:
        train_data_samples = minutes_to_samples(args.train_data_min, step_ms=trial_duration_ms)
        subset = Subset(train_loader.dataset, range(min(train_data_samples, len(train_loader.dataset))))
        train_loader = torch.utils.data.DataLoader(
            subset, batch_size=train_loader.batch_size, shuffle=False,
            num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
        )

    # test_loader's batch_size ALSO needs capping to its own dataset
    # size, same as train_loader can be -- confirmed directly against a
    # real crash: create_dataloaders() applies the SAME args.batch_size
    # to both loaders, but test has a fundamentally different, much
    # smaller trial count (often exactly 1, for the whole-trial grouped
    # datasets elsewhere in this project) -- batch_size > that count
    # produced ZERO test batches in the real run. Confirmed directly
    # (not assumed) that PyTorch's own DataLoader DEFAULT
    # (drop_last=False) does NOT reproduce this on its own -- meaning
    # create_dataloaders() itself must set drop_last=True somewhere not
    # visible here. Rebuilding with drop_last=False EXPLICITLY below,
    # not relying on whatever the default happens to be, since that's
    # exactly what avoids the zero-batch outcome regardless of what the
    # original loader's own setting was. Test evaluation should never
    # drop any of its own data anyway.
    if test_loader.batch_size > len(test_loader.dataset):
        safe_test_batch_size = len(test_loader.dataset)
        print(f"test_loader's batch_size ({test_loader.batch_size}) exceeds its own dataset "
              f"size ({len(test_loader.dataset)}) -- capping to {safe_test_batch_size} to avoid "
              f"zero test batches.")
        test_loader = torch.utils.data.DataLoader(
            test_loader.dataset, batch_size=safe_test_batch_size, shuffle=False, drop_last=False,
            num_workers=test_loader.num_workers, pin_memory=test_loader.pin_memory,
        )

    # A POOLED test set (e.g. one whole trial per session, sessions of
    # genuinely different recording durations) can have samples of
    # different lengths even when the count itself is fine. Batching
    # different-length trials together fails (PyTorch's default
    # collate_fn can't stack mismatched shapes into one tensor) --
    # confirmed directly this is a real, not theoretical, risk: every
    # grouped dataset's train split was already made explicitly uniform
    # (--discard-train-remainder) for exactly this reason, but nothing
    # equivalent exists for a POOLED test split, since each session's
    # test trial keeps its own, natural length. Checked using the same
    # test_loader.dataset[i][1].shape[0] pattern already used above (see
    # trial_n_timesteps) for consistency, not a new convention.
    if test_loader.batch_size > 1:
        trial_lengths = {test_loader.dataset[i][1].shape[0] for i in range(len(test_loader.dataset))}
        if len(trial_lengths) > 1:
            print(f"test set has {len(trial_lengths)} distinct trial lengths "
                  f"({sorted(trial_lengths)}) -- forcing test batch_size to 1, since "
                  f"batching different-length trials together isn't possible.")
            test_loader = torch.utils.data.DataLoader(
                test_loader.dataset, batch_size=1, shuffle=False, drop_last=False,
                num_workers=test_loader.num_workers, pin_memory=test_loader.pin_memory,
            )

    # Catch a zero-batch loader HERE, immediately and clearly, rather
    # than downstream as a confusing ZeroDivisionError inside
    # train_epoch()/evaluate_epoch() -- confirmed directly this
    # otherwise happens with no other indication of the real cause.
    if len(train_loader) == 0:
        raise ValueError(f"train_loader has zero batches ({len(train_loader.dataset)} samples, "
                          f"batch_size={train_loader.batch_size}) -- nothing to train on.")
    if len(test_loader) == 0:
        raise ValueError(f"test_loader has zero batches ({len(test_loader.dataset)} samples, "
                          f"batch_size={test_loader.batch_size}) -- nothing to evaluate on.")

    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

    if args.wandb:
        wandb.config.update({
            "train_samples": len(train_loader.dataset), "test_samples": len(test_loader.dataset),
            "train_batches": len(train_loader), "test_batches": len(test_loader),
        })

    model = create_model(
        use_spikingjelly=args.use_spikingjelly,
        last_layer_reset=args.last_layer_reset,
        spike_fn=(sinabs.activation.MultiSpike if args.spike_fn == "multi" else sinabs.activation.SingleSpike),
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
        tau_syn=None,
        velocity_lo=args.velocity_lo,
        velocity_hi=args.velocity_hi,
        velocity_margin=args.velocity_margin,
    ).to(device)

    if args.resume and args.init_weights_from:
        raise ValueError("--resume and --init-weights-from are mutually exclusive -- "
                          "--resume continues an interrupted run (restores optimizer state, "
                          "epoch count, best_loss); --init-weights-from starts a genuinely "
                          "fresh training run using only another checkpoint's weights as the "
                          "starting point. Pick one.")

    if args.init_weights_from:
        # Fine-tuning, NOT resume -- see this flag's own --help text for
        # the full reasoning. Loads weights BEFORE the optimizer is
        # constructed below, so the optimizer's own parameter references
        # start from these loaded values, not the freshly-initialized
        # ones create_model() produced above.
        source_checkpoint = torch.load(args.init_weights_from, map_location=device)
        load_model_weights(model, source_checkpoint['model_state_dict'],
                            neuron_type=args.neuron_type,
                            source_description=args.init_weights_from)
        print(f"Initialized model weights from {args.init_weights_from} "
              f"(fresh optimizer/epoch-count/best_loss follow, this is fine-tuning, not resume)")
        if args.wandb:
            wandb.config.update({"init_weights_from": args.init_weights_from})

    if args.verbose:
        print(f"\nModel architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        if args.wandb:
            wandb.config.update({"total_params": total_params})

    if args.wandb:
        wandb.watch(model, criterion=nn.MSELoss(), log="all", log_freq=100)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                   patience=args.scheduler_patience, min_lr=args.min_lr)
    early_stopping = None if args.disable_early_stopping else EarlyStopping(patience=args.patience)

    start_epoch = 0
    best_loss = float("inf")
    train_losses, test_losses, lr_history = [], [], []
    lr_reduction_count = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        train_losses = checkpoint.get("train_losses", [])
        test_losses = checkpoint.get("test_losses", [])
        lr_history = checkpoint.get("lr_history", [])
        lr_reduction_count = checkpoint.get("lr_reduction_count", 0)
        print(f"Resumed from epoch {start_epoch} with {lr_reduction_count} LR reductions")
        if args.wandb:
            wandb.config.update({"resumed_from": args.resume, "start_epoch": start_epoch})
        if start_epoch >= args.epochs:
            # Confirmed directly this crashes otherwise: for epoch in
            # range(start_epoch, args.epochs) is an empty loop when
            # start_epoch >= args.epochs, leaving `epoch` never bound in
            # this scope -- the code after the loop (saving the "final"
            # checkpoint) then raises a cryptic UnboundLocalError instead
            # of explaining what actually went wrong. Failing loudly and
            # clearly HERE, before that happens, with the actual fix
            # spelled out -- this is a real, not hypothetical, situation:
            # it happens whenever a checkpoint saved under one --epochs
            # value is resumed under an --epochs that hasn't been raised
            # to account for it.
            raise ValueError(
                f"Resumed checkpoint's own epoch ({start_epoch}) is already >= "
                f"--epochs ({args.epochs}) -- there is nothing left to train under this "
                f"budget. To continue training further, raise --epochs beyond "
                f"{start_epoch} (e.g. to {start_epoch + args.epochs}) and resubmit; "
                f"the checkpoint itself is fine, only the epoch budget needs increasing.")

    print("\nStarting training...")
    training_start_time = time.time()
    scale_kwargs = dict(v_lo=args.velocity_lo, v_hi=args.velocity_hi,
                         v_margin=args.velocity_margin, loss_eps=args.loss_eps)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            args.use_amp, args.log_interval, args.wandb, args.verbose, **scale_kwargs,
            channel_lasso_weight=args.channel_lasso_weight,
        )
        train_losses.append(train_loss)

        test_loss, test_metrics = evaluate_epoch(
            model, test_loader, criterion, device, epoch, args.wandb, args.verbose,
            **scale_kwargs,
        )
        test_losses.append(test_loss)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            lr_reduction_count += 1
            print(f"\n*** Learning rate reduced from {old_lr:.6f} to {new_lr:.6f} "
                  f"(reduction #{lr_reduction_count}) ***\n")
            if args.wandb:
                wandb.log({"lr_reduction_event": lr_reduction_count, "old_lr": old_lr,
                           "new_lr": new_lr, "lr_reduction_epoch": epoch + 1}, step=epoch)
        lr_history.append(new_lr)

        epoch_time = time.time() - epoch_start_time
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_membrane_mean": train_metrics["membrane_mean"],
                "train/epoch_membrane_max": train_metrics["membrane_max"],
                "train/epoch_non_zero_ratio": train_metrics["non_zero_ratio"],
                "train/epoch_spike_rate": train_metrics["spike_rate"],
                "test/epoch_loss": test_loss,
                "test/epoch_velocity_error": test_metrics["velocity_error"],
                "test/epoch_membrane_mean": test_metrics["membrane_mean"],
                "test/epoch_membrane_max": test_metrics["membrane_max"],
                "test/epoch_non_zero_ratio": test_metrics["non_zero_ratio"],
                "test/epoch_spike_rate": test_metrics["spike_rate"],
                "learning_rate": train_metrics["learning_rate"],
                "epoch_time": epoch_time,
            }, step=epoch)

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary (Time: {epoch_time:.1f}s)")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
        print(f"  Velocity Error: {test_metrics['velocity_error']:.5f}")
        print(f"  Membrane - Mean: {test_metrics['membrane_mean']:.3f}, Max: {test_metrics['membrane_max']:.3f}")
        print(f"  Non-zero ratio: {test_metrics['non_zero_ratio']:.2%}  (spike rate: {test_metrics['spike_rate']:.4f} avg spikes/neuron/timestep)")
        print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
        print(f"{'='*80}\n")

        if test_loss < best_loss:
            best_loss = test_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model_weights.pth")
            save_checkpoint(
                model, optimizer, epoch, test_loss, best_path,
                additional_info={
                    "input_shape": input_shape, "best_loss": best_loss,
                    "train_losses": train_losses, "test_losses": test_losses,
                    "lr_history": lr_history, "lr_reduction_count": lr_reduction_count,
                    "args": vars(args), "is_best": True,
                },
            )
            print(f"Saved best model with loss {best_loss:.5f}")
            if args.wandb:
                wandb.save(best_path)
                wandb.run.summary["best_test_loss"] = best_loss
                wandb.run.summary["best_epoch"] = epoch

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{exp_name}_epoch{epoch+1}.pth")
            save_checkpoint(
                model, optimizer, epoch, test_loss, checkpoint_path,
                additional_info={
                    "best_loss": best_loss, "train_losses": train_losses, "test_losses": test_losses,
                    "lr_history": lr_history, "lr_reduction_count": lr_reduction_count, "args": vars(args),
                },
            )
            if args.wandb:
                wandb.save(checkpoint_path)

        if args.disable_early_stopping:
            if optimizer.param_groups[0]["lr"] <= args.min_lr:
                print(f"\nReached minimum learning rate ({args.min_lr}) at epoch {epoch+1}")
                if args.wandb:
                    wandb.run.summary["reached_min_lr"] = True
                    wandb.run.summary["min_lr_epoch"] = epoch
                if lr_reduction_count >= args.max_lr_reductions:
                    print(f"Reached maximum LR reductions ({args.max_lr_reductions}). Stopping training.")
                    break
        else:
            if early_stopping and early_stopping(test_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                if args.wandb:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stop_epoch"] = epoch
                break

    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best test loss: {best_loss:.5f}")

    final_path = os.path.join(args.checkpoint_dir, f"final_model_{exp_name}.pth")
    save_checkpoint(
        model, optimizer, epoch, test_loss, final_path,
        additional_info={
            "input_shape": input_shape, "best_loss": best_loss,
            "train_losses": train_losses, "test_losses": test_losses,
            "args": vars(args), "is_final": True,
        },
    )

    plot_path = os.path.join(args.checkpoint_dir, f"training_curves_{exp_name}.png")
    plot_training_curves(train_losses, test_losses, plot_path,
                          title=f"Training Progress - {exp_name}", lr_history=lr_history)

    if args.wandb:
        wandb.run.summary.update({
            "total_training_time": training_time, "final_train_loss": train_loss,
            "final_test_loss": test_loss, "final_velocity_error": test_metrics["velocity_error"],
            "total_epochs": epoch + 1 - start_epoch, "total_lr_reductions": lr_reduction_count,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
        })
        wandb.save(final_path)
        wandb.log({"training_curves": wandb.Image(plot_path)}, step=epoch)

    summary_path = os.path.join(args.checkpoint_dir, f"summary_{exp_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Final epoch: {epoch+1}\n")
        f.write(f"Best test loss: {best_loss:.5f}\n")
        f.write(f"Training time: {training_time/60:.1f} minutes\n")
        f.write(f"Arguments: {vars(args)}\n")

    print(f"\nResults saved to {args.checkpoint_dir}")

    if args.wandb:
        print(f"WandB run URL: {wandb.run.get_url()}")
        wandb.finish()


if __name__ == "__main__":
    main()
