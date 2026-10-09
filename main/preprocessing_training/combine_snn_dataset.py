"""
Builds a dataset of LARGE, multi-base-window TRAIN trials from the
SNN's pre-windowing h5 source -- group_size consecutive base windows
(default 256, i.e. 256*65 = 16,640 timesteps = ~66.6s per trial)
concatenated into one trial, saved as consecutive {i}.pkl files under a
train/ subdirectory. Remainder windows that don't fill a complete group
are NOT discarded -- saved as one final, shorter trial instead.

TEST is handled differently -- saved as ONE SINGLE, WHOLE, unchunked
trial regardless of group_size, not split into group_size pieces the
way train is. See the comment directly above where main() saves it for
the two real reasons (a statistical one about how per-trial-averaged
metrics can be skewed by one disproportionately short trial, and a
memory one about evaluation never needing a backward-pass graph at all).

Motivation, and why this is a genuinely different approach from
detach_states()-based cross-trial state continuity (which this project
already tried twice, as "continuous" and "chunked" training modes, and
removed after both underperformed windowed training -- see
train_bmi.py's own docstring): this script changes ONLY the dataset's
own trial granularity. Every trial produced here is still self-
contained and gets exactly one reset_state=True at its start, run
through the completely UNMODIFIED windowed training loop -- no
detach_states(), no cross-trial state carrying, no training-loop
changes of any kind. The goal is more temporal context PER TRIAL
(16,640 timesteps instead of 260) without touching the mechanism that
already failed twice.

Real, separate risk this does NOT eliminate, confirmed directly rather
than assumed away: probe_bptt_memory.py measured real memory scaling
for one full, UNTRUNCATED forward+backward pass at increasing sequence
lengths on the actual architecture -- group_size=256 (16,640 timesteps)
cost about 1.25GB for that graph alone, well within a 16GB budget with
real headroom, but this has NOT been validated against a real training
run end to end. Confirm on your own system before committing to a full
sweep, especially if group_size is increased beyond what
probe_bptt_memory.py was actually run against.

Reuses build_continuous_trials.py's load_and_verify_continuous_source()
(same stride-verification, y_trace-vs-y_end disambiguation, and
reconstruction logic, imported directly -- not reimplemented a second
time) and test_all_decoders.py's compute_aligned_split() for the
train/test boundary, exactly as build_continuous_trials.py does.

CLI usage:
    python combine_snn_dataset.py \
        --h5-path /users/bfalkenb/data/bfalkenb/data/dataset/mua/indy_20160407_02_snn.h5 \
        --session-id indy_20160407_02 \
        --group-size 8 \
        --test-frac 0.1 \
        --output-dir datasets/bmi/mua_8_group
"""

import argparse
import os
import pickle as pkl
import sys

import numpy as np

# test_all_decoders.py (for compute_aligned_split) now lives in
# visualization/, not this file's own directory (preprocessing_training/)
# -- both anchored to BMI_PROJECT_ROOT rather than assumed to be a fixed
# number of directories away, matching single_subject_pipeline.py's own
# convention. build_continuous_trials.py is assumed to still be a sibling
# of THIS file (not independently verified -- see module docstring).
_BMI_PROJECT_ROOT = os.environ.get(
    "BMI_PROJECT_ROOT", "/users/bfalkenb/spike_bmi_main")
sys.path.insert(0, os.path.join(_BMI_PROJECT_ROOT, "visualization"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_continuous_trials import load_and_verify_continuous_source, BASE_NPERSEG, STEP_TIME_S
from test_all_decoders import compute_aligned_split


def _save_chunk(split_dir, chunk_idx, input_spikes, velocity):
    output_path = os.path.join(split_dir, f"{chunk_idx}.pkl")
    with open(output_path, "wb") as f:
        pkl.dump({
            "input_spikes": input_spikes.astype(np.float32),
            "velocity": velocity.astype(np.float32),
        }, f)
    duration_s = velocity.shape[0] * STEP_TIME_S
    print(f"  {output_path}: input_spikes shape={input_spikes.shape}, "
          f"velocity shape={velocity.shape} ({duration_s:.1f}s)")


def chunk_and_save(output_dir, split_name, input_spikes_full, velocity_full, group_size,
                    discard_remainder=False):
    """input_spikes_full: (C, T), velocity_full: (T, 2) -- one region's
    full continuous data. Splits into consecutive, non-overlapping
    chunks of group_size*BASE_NPERSEG timesteps each, saving each as its
    own {i}.pkl. The FINAL chunk gets whatever's left over (shorter than
    the rest, if total length isn't an exact multiple of the chunk
    length) rather than being discarded by default -- only skipped
    entirely if the remainder is exactly 0, OR if discard_remainder=True
    is explicitly requested.

    discard_remainder exists specifically to make every saved trial the
    SAME length -- batch_size=1 has been required everywhere in this
    project for the grouped dataset specifically because trials aren't
    uniform length (this one shorter remainder chunk per session is the
    reason why). With every trial forced to the same chunk_len, real
    batching (batch_size>1) becomes possible, which can meaningfully cut
    per-timestep Python/dispatch overhead -- a different, and likely
    more impactful, lever than simply not computing the (already tiny,
    confirmed under 0.5% of total compute) remainder chunk at all.
    """
    chunk_len = group_size * BASE_NPERSEG
    total_len = input_spikes_full.shape[1]
    n_full_chunks = total_len // chunk_len
    remainder = total_len - n_full_chunks * chunk_len

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    remainder_note = " (no remainder)" if remainder == 0 else (
        f", discarding the {remainder}-timestep remainder (--discard-train-remainder)"
        if discard_remainder else f", plus 1 shorter trailing chunk of {remainder} timesteps")
    print(f"\n{split_name}: {total_len} total timesteps -> {n_full_chunks} full chunks of "
          f"{chunk_len} timesteps each" + remainder_note)

    chunk_idx = 0
    for i in range(n_full_chunks):
        start = i * chunk_len
        end = start + chunk_len
        _save_chunk(split_dir, chunk_idx, input_spikes_full[:, start:end], velocity_full[start:end, :])
        chunk_idx += 1

    if remainder > 0 and not discard_remainder:
        start = n_full_chunks * chunk_len
        _save_chunk(split_dir, chunk_idx, input_spikes_full[:, start:], velocity_full[start:, :])
        chunk_idx += 1

    return chunk_idx


def main(args):
    X_continuous_full, y_continuous_full, n_windows = load_and_verify_continuous_source(args.h5_path)
    if X_continuous_full is None:
        return

    total_raw_samples_full = n_windows * BASE_NPERSEG
    N = total_raw_samples_full - BASE_NPERSEG
    n_train, n_test = compute_aligned_split(N, args.test_frac, base_nperseg=BASE_NPERSEG)
    assert n_train % BASE_NPERSEG == 0, "compute_aligned_split()'s own guarantee -- should never fail"
    print(f"\ncompute_aligned_split(N={N}, test_frac={args.test_frac}) -> "
          f"n_train={n_train} samples, n_test={n_test} samples")

    train_input_spikes = X_continuous_full[:, :n_train]
    train_velocity = y_continuous_full[:n_train, :]
    test_input_spikes = X_continuous_full[:, n_train:]
    test_velocity = y_continuous_full[n_train:, :]

    output_dir = os.path.join(args.output_dir, args.session_id)
    n_train_chunks = chunk_and_save(output_dir, "train", train_input_spikes, train_velocity,
                                     args.group_size, discard_remainder=args.discard_train_remainder)

    # Test is saved as ONE WHOLE trial, deliberately NOT chunked by
    # group_size the way train is -- two real reasons, not just
    # convenience. First, statistical: test_frac=0.1 means the test
    # region is already far shorter than train (confirmed: ~1-2 minutes
    # vs ~12+ here), so chunking it the same way produces very few test
    # trials, one of which (the group_size remainder) can end up much
    # shorter than the others -- if evaluation code averages metrics
    # PER TRIAL rather than weighting by trial length, that short trial
    # gets the same vote as a much longer one, and if it happens to be
    # an "easy" stretch, it can pull the averaged test metric down in a
    # way that doesn't reflect overall performance. One single, whole
    # test trial removes this entirely -- nothing to unevenly average
    # over. Second, memory: unlike train, evaluation runs under
    # torch.no_grad() (confirmed directly in train_bmi.py's
    # evaluate_epoch()) -- no backward-pass graph is ever retained, so
    # the memory concern that motivated chunking TRAIN in the first
    # place doesn't apply here in the same way; the test region is also
    # roughly 1/10th the length of train to begin with.
    if n_test > 0:
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        _save_chunk(test_dir, 0, test_input_spikes, test_velocity)
        n_test_chunks = 1
    else:
        n_test_chunks = 0
    print(f"\nDone: {n_train_chunks} train trial(s), {n_test_chunks} test trial(s) under "
          f"{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=256,
                         help="Base windows (each 65 timesteps / 260ms) per TRAIN trial. "
                              "Default 256 -> 16,640 timesteps (~66.6s) per trial. Test is "
                              "always saved as one whole, unchunked trial regardless of this "
                              "value -- see the comment above main()'s test-saving branch for "
                              "why.")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="datasets/bmi/mua_huge")
    parser.add_argument("--discard-train-remainder", action="store_true",
                         help="Drop each session's shorter, trailing remainder train trial "
                              "instead of saving it -- makes every saved train trial EXACTLY "
                              "group_size*65 timesteps long, enabling batch_size>1 (previously "
                              "impossible with mixed trial lengths). Confirmed directly earlier "
                              "that the remainder itself is under 0.5%% of one session's total "
                              "train compute -- this flag isn't primarily about that; it's about "
                              "enabling real batching, a different and likely more impactful "
                              "lever. Test is NEVER affected by this flag -- always saved as one "
                              "whole, unchunked trial regardless (see main()'s own comment for "
                              "why discarding anything from test would reintroduce a real risk).")
    args = parser.parse_args()
    main(args)
