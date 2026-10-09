"""
Directly compares the SET of trial IDs present in raw_trials/ vs
ann_windowed_per_trial/ for one session -- built to isolate an exact
discrepancy (some trials genuinely missing an ANN file vs. some ANN
files existing but not being matched) rather than keep reasoning from
aggregate counts alone.

Usage:
    python diagnose_ann_trial_id_mismatch.py \
        --output-root /users/bfalkenb/scratch/bfalkenb/data/hkm_purgatory \
        --session sub-Jenkins_ses-20090918_behavior+ecephys
"""
import argparse
import glob
import os
import re
from collections import Counter

RAW_PATTERN = re.compile(r"_trial(\d+)")
ANN_PATTERN = re.compile(r"_trial(\d+)_\w+\.h5$")


def get_trial_ids(directory, pattern):
    """Returns id_to_files -- maps trial_id -> list of filenames that
    matched it (so a trial_id matched by more than one file is visible
    directly, not silently overwritten)."""
    id_to_files = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.h5"))):
        basename = os.path.basename(path)
        m = pattern.search(basename)
        if not m:
            print(f"  [no match] {basename} -- doesn't match the expected pattern at all")
            continue
        trial_id = int(m.group(1))
        id_to_files.setdefault(trial_id, []).append(basename)
    return id_to_files


def main(args):
    raw_dir = os.path.join(args.output_root, "raw_trials", args.session)
    ann_dir = os.path.join(args.output_root, "ann_windowed_per_trial", args.session)

    raw_ids = get_trial_ids(raw_dir, RAW_PATTERN)
    ann_ids = get_trial_ids(ann_dir, ANN_PATTERN)

    raw_set = set(raw_ids.keys())
    ann_set = set(ann_ids.keys())

    print(f"\n{'='*72}\n{args.session}\n{'='*72}")
    print(f"  raw_trials:            {len(raw_set)} distinct trial IDs, "
          f"{sum(len(v) for v in raw_ids.values())} files total")
    print(f"  ann_windowed_per_trial: {len(ann_set)} distinct trial IDs, "
          f"{sum(len(v) for v in ann_ids.values())} files total")

    # Trial IDs matched by MORE than one file -- these would silently
    # inflate the file count above the distinct-ID count, and are worth
    # seeing directly rather than assumed away.
    ann_dupes = {tid: files for tid, files in ann_ids.items() if len(files) > 1}
    if ann_dupes:
        print(f"\n  {len(ann_dupes)} trial ID(s) matched by MULTIPLE ann files:")
        for tid, files in sorted(ann_dupes.items())[:10]:
            print(f"    trial {tid}: {files}")
        if len(ann_dupes) > 10:
            print(f"    ... and {len(ann_dupes) - 10} more")

    missing_in_ann = raw_set - ann_set
    print(f"\n  Trial IDs in raw_trials but with NO ann file at all: {len(missing_in_ann)}")
    if missing_in_ann:
        sample = sorted(missing_in_ann)[:10]
        print(f"    First 10: {sample}")
        # Check if missing IDs cluster at a particular range -- e.g. all
        # above some threshold, which would suggest windowing simply
        # stopped partway through, rather than skipping scattered trials.
        print(f"    Range of ALL missing IDs: [{min(missing_in_ann)}, {max(missing_in_ann)}]")
        print(f"    Range of ALL raw IDs:     [{min(raw_set)}, {max(raw_set)}]")

    extra_in_ann = ann_set - raw_set
    if extra_in_ann:
        print(f"\n  WARNING: {len(extra_in_ann)} trial ID(s) exist in ann_windowed_per_trial "
              f"but NOT in raw_trials at all (shouldn't be possible if ANN windowing reads "
              f"FROM raw_trials): {sorted(extra_in_ann)[:10]}")

    matched = raw_set & ann_set
    print(f"\n  Trial IDs present in BOTH (this is what actually gets compared): {len(matched)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    args = parser.parse_args()
    main(args)
