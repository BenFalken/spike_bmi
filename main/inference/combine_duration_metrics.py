"""
Combines the per-session, per-duration metrics files written by
run_duration_sweep_eval_array.sbatch's parallel array tasks into ONE
combined_metrics.json with the EXACT same structure
test_all_decoders.py --multi_session --combined_metrics_path would have
produced directly: {session_id: {duration_tag: {decoder: metrics}}}.

Necessary, not just convenient: each individual per-duration file
(test_all_decoders.py's --metrics_save_path, auto-suffixed per duration
via _suffixed_path()) is NOT the bare {decoder: metrics} dict --
run_session() wraps it:
    {'session': ..., 'duration_tag': ..., 'train_duration_minutes': ...,
     'start_raw': ..., 'end_raw': ..., 'n_samples': ...,
     'decoders': [...], 'metrics': {decoder: metrics}}
The actual per-decoder metrics live under the 'metrics' key -- reading a
file's top level directly (skipping that unwrap) would silently produce
a combined_metrics.json decoder_comparison_4x2.py can't read correctly.

Usage:
    python combine_duration_metrics.py \
        --results-dir /users/bfalkenb/scratch/bfalkenb/data/results/test_all_decoders/bmi/loco \
        --output /users/bfalkenb/scratch/bfalkenb/data/results/test_all_decoders/bmi/loco/combined_metrics_durations.json
"""

import argparse
import glob
import json
import os
import re


def combine(results_dir, output_path):
    # Matches {session_id}_metrics_{duration_tag}.json -- e.g.
    # indy_20160407_02_metrics_5min.json. Deliberately does NOT match the
    # unsuffixed {session_id}_metrics.json (a single, non-swept run, if
    # one happens to exist in the same directory) -- this script is
    # specifically for the duration-sweep case.
    pattern = re.compile(r"^(?P<session_id>.+)_metrics_(?P<duration_tag>[\d.]+min)\.json$")

    combined = {}
    files = sorted(glob.glob(os.path.join(results_dir, "*_metrics_*min.json")))
    if not files:
        raise FileNotFoundError(
            f"No *_metrics_*min.json files found under {results_dir} -- "
            f"has run_duration_sweep_eval_array.sbatch actually finished any tasks yet?")

    n_loaded = 0
    for filepath in files:
        basename = os.path.basename(filepath)
        m = pattern.match(basename)
        if not m:
            print(f"[skip] {basename}: doesn't match the expected "
                  f"{{session_id}}_metrics_{{duration_tag}}.json pattern")
            continue
        session_id = m.group("session_id")
        duration_tag = m.group("duration_tag")

        with open(filepath, "r") as f:
            file_content = json.load(f)

        if "metrics" not in file_content:
            print(f"[skip] {basename}: no 'metrics' key found -- unexpected file structure, "
                  f"not test_all_decoders.py's own wrapped format")
            continue

        combined.setdefault(session_id, {})[duration_tag] = file_content["metrics"]
        n_loaded += 1

    if not combined:
        raise RuntimeError(f"Found {len(files)} file(s) under {results_dir}, but none matched "
                            f"the expected pattern/structure -- nothing to combine.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    n_sessions = len(combined)
    durations_per_session = {sid: sorted(d.keys()) for sid, d in combined.items()}
    incomplete = {sid: durs for sid, durs in durations_per_session.items()
                  if len(durs) < max(len(d) for d in durations_per_session.values())}
    print(f"Combined {n_loaded} file(s) across {n_sessions} session(s) -> {output_path}")
    if incomplete:
        print(f"\n[note] {len(incomplete)} session(s) have FEWER durations than the max found "
              f"-- likely still-running or failed array tasks, not necessarily an error:")
        for sid, durs in incomplete.items():
            print(f"  {sid}: {durs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    combine(args.results_dir, args.output)
