"""
Combines the per-session FULL-MODEL (non-duration) metrics files written
by run_test_all_decoders_array.sbatch's full-model call (the invocation
WITHOUT --train_durations) into ONE combined_metrics.json with the flat
structure decoder_comparison_4x2.py's rows 0/1/3 actually read:
{session_id: {decoder: metrics}}.

Companion to combine_duration_metrics.py, not a replacement -- that
script combines the DURATION-TAGGED files (*_metrics_*min.json) into the
nested {session_id: {duration_tag: {decoder: metrics}}} structure row 2
needs. Neither script covers what the other does; run both, into their
own separate output files (decoder_comparison_4x2.py's
--combined-metrics-path and --duration-metrics-path respectively).

Each individual full-model file (test_all_decoders.py's own
--metrics_save_path, UNSUFFIXED since duration_tag is None for this
call) is NOT the bare {decoder: metrics} dict -- confirmed directly
against test_all_decoders.py's own save code, which wraps it
UNCONDITIONALLY, for both the duration-swept and full-model cases alike:
    {'session': ..., 'duration_tag': None, 'train_duration_minutes': None,
     'start_raw': ..., 'end_raw': ..., 'n_samples': ...,
     'decoders': [...], 'metrics': {decoder: metrics}}
The actual per-decoder metrics live under the 'metrics' key -- reading a
file's top level directly (skipping that unwrap) would silently produce
a combined_metrics.json decoder_comparison_4x2.py can't read correctly.

Matches ONLY the bare {session_id}_metrics.json pattern -- deliberately
excludes {session_id}_metrics_{N}min.json (that's
combine_duration_metrics.py's own job), via a regex that requires the
filename to end immediately after "_metrics.json", not just start with
"..._metrics".

Usage:
    python combine_full_metrics.py \
        --results-dir /users/bfalkenb/scratch/bfalkenb/data/results/test_all_decoders/bmi/loco \
        --output /users/bfalkenb/scratch/bfalkenb/data/results/test_all_decoders/bmi/loco/combined_metrics.json
"""

import argparse
import glob
import json
import os
import re


def combine(results_dir, output_path):
    # Matches {session_id}_metrics.json exactly -- NOT
    # {session_id}_metrics_5min.json (that's a different file, handled
    # by combine_duration_metrics.py instead). The `$` anchor after
    # "_metrics.json" is what enforces this: "_metrics_5min.json" does
    # NOT end there, so it correctly falls through to no-match.
    pattern = re.compile(r"^(?P<session_id>.+)_metrics\.json$")

    combined = {}
    all_metrics_files = sorted(glob.glob(os.path.join(results_dir, "*_metrics.json")))
    # glob's own "*_metrics.json" is NOT enough by itself to exclude
    # "..._metrics_5min.json" -- that filename also ends in a different
    # place ("5min.json"), so glob's wildcard never matches it in the
    # first place; the extra regex re-check below is a second, explicit
    # confirmation of the exact boundary, not a redundant no-op.
    files = [f for f in all_metrics_files if pattern.match(os.path.basename(f))]

    if not files:
        raise FileNotFoundError(
            f"No bare {{session_id}}_metrics.json files found under {results_dir} -- "
            f"has run_test_all_decoders_array.sbatch's full-model call (the one WITHOUT "
            f"--train_durations) actually finished any sessions yet?")

    n_loaded = 0
    for filepath in files:
        basename = os.path.basename(filepath)
        m = pattern.match(basename)
        session_id = m.group("session_id")

        with open(filepath, "r") as f:
            file_content = json.load(f)

        if "metrics" not in file_content:
            print(f"[skip] {basename}: no 'metrics' key found -- unexpected file structure, "
                  f"not test_all_decoders.py's own wrapped format")
            continue

        combined[session_id] = file_content["metrics"]
        n_loaded += 1

    if not combined:
        raise RuntimeError(f"Found {len(files)} file(s) under {results_dir}, but none matched "
                            f"the expected structure -- nothing to combine.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Combined {n_loaded} full-model session file(s) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    combine(args.results_dir, args.output)
