"""
Short diagnostic: checks every train/*.pkl trial in a session's mua_N_group
directory for uniform length -- confirms/locates the RuntimeError from
DataLoader's default collate_fn ("stack expects each tensor to be equal
size").

Usage:
    python diagnose_trial_lengths.py \
        --session-dir /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/indy/mua_8_group/indy_20160407_02
"""
import argparse
import glob
import os
import pickle as pkl
from collections import Counter


def main(args):
    train_dir = os.path.join(args.session_dir, "train")
    files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")),
                    key=lambda f: int(os.path.basename(f).split(".")[0]))
    if not files:
        raise SystemExit(f"No .pkl files found under {train_dir}")

    lengths = {}
    for fpath in files:
        with open(fpath, "rb") as f:
            data = pkl.load(f)
        lengths[os.path.basename(fpath)] = data["input_spikes"].shape[-1]

    length_counts = Counter(lengths.values())
    print(f"{len(files)} train trials, lengths: {dict(sorted(length_counts.items()))}")

    if len(length_counts) == 1:
        print("OK: every trial has the same length -- discard-train-remainder was "
              "applied correctly for this session.")
        return

    modal_length = length_counts.most_common(1)[0][0]
    print(f"\nNOT uniform -- {modal_length} is the modal (expected group_size*65) length. "
          f"Off-length trials:")
    for fname, length in lengths.items():
        if length != modal_length:
            print(f"  {fname}: {length} timesteps ({length / 65:.2f} base windows)")
    print(f"\nThis confirms --discard-train-remainder was NOT applied when this "
          f"session's dataset was built -- rebuild it (see the fix to "
          f"build_combined_snn_dataset() in single_subject_pipeline.py) rather "
          f"than working around it downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=str, required=True,
                         help="A session's own mua_N_group directory, e.g. "
                              ".../mua_8_group/indy_20160407_02")
    args = parser.parse_args()
    main(args)
