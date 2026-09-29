"""
Builds a pooled, multi-session dataset directory for pretraining, by
SYMLINKING (not copying -- no data duplication) every session's
train/test .pkl files from an existing mua_large-style dataset root
into one unified directory, renamed to avoid filename collisions
(every session's own train/ folder independently starts numbering at
0.pkl, 1.pkl, ...).

METHODOLOGICAL NOTE, worth reading before using this for the workflow
this was built for (pretrain on many sessions, fine-tune per-session,
evaluate per-session): if a session's OWN data is included in this
pool, and that same session is later the fine-tuning target, its test
split gets indirectly "seen" during the PRETRAINING phase's own
checkpoint selection (best_loss tracking) -- even though it never
drives a gradient update directly, this is a real form of leakage
against a genuinely single-session evaluation standard. Use
--exclude-session to build a leave-one-session-out pool for a specific
fine-tuning target instead -- more expensive (a separate pretrained
checkpoint per session, rather than one shared one), but methodologically
clean: that session's data is never touched until final evaluation.

CLI usage:
    # Pool everything (cheaper, some indirect leakage into pretraining's
    # own checkpoint selection for whichever sessions get fine-tuned later):
    python build_pretraining_pool.py \
        --mua-large-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/loco/mua_8_group \
        --output-dir /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/loco/mua_pretrain_pool
"""

import argparse
import os


def pool_split(mua_large_root, session_ids, split_name, output_dir):
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    n_linked = 0
    n_sessions_included = 0
    for session_id in session_ids:
        split_dir = os.path.join(mua_large_root, session_id, split_name)
        if not os.path.isdir(split_dir):
            print(f"  [skip] {session_id}: no {split_name}/ subfolder")
            continue

        pkl_files = sorted(f for f in os.listdir(split_dir) if f.endswith('.pkl'))
        if not pkl_files:
            print(f"  [skip] {session_id}: {split_name}/ has no .pkl files")
            continue

        for fname in pkl_files:
            source_path = os.path.abspath(os.path.join(split_dir, fname))
            # Renamed to include the session id -- every session's own
            # split/ folder independently starts at 0.pkl, so pooling
            # without renaming would silently overwrite/collide.
            link_name = f"{session_id}_{fname}"
            link_path = os.path.join(split_output_dir, link_name)
            if os.path.lexists(link_path):
                # Only reached if this script is re-run over a partially-built
                # pool -- remove and relink rather than silently skip, so a
                # changed source file is always reflected, not stale.
                os.remove(link_path)
            os.symlink(source_path, link_path)
            n_linked += 1
        n_sessions_included += 1

    print(f"{split_name}: linked {n_linked} files from {n_sessions_included} session(s) "
          f"into {split_output_dir}")
    return n_linked, n_sessions_included


def main(args):
    all_session_ids = sorted(
        d for d in os.listdir(args.mua_large_root)
        if os.path.isdir(os.path.join(args.mua_large_root, d))
    )
    if not all_session_ids:
        raise ValueError(f"No session directories found under {args.mua_large_root}")

    excluded = set(args.exclude_session or [])
    unknown_excludes = excluded - set(all_session_ids)
    if unknown_excludes:
        raise ValueError(f"--exclude-session value(s) not found under {args.mua_large_root}: "
                          f"{sorted(unknown_excludes)} -- available sessions: {all_session_ids}")

    session_ids = [s for s in all_session_ids if s not in excluded]
    print(f"Found {len(all_session_ids)} session(s) total under {args.mua_large_root}")
    if excluded:
        print(f"Excluding {len(excluded)} session(s): {sorted(excluded)}")
    print(f"Pooling {len(session_ids)} session(s): {session_ids}\n")

    # Fail clearly and immediately, not by silently building an empty
    # pool that only surfaces the problem much later as a confusing
    # crash deep inside PyTorch's own DataLoader/RandomSampler
    # (num_samples=0) -- confirmed directly, this is exactly what
    # happened before this check existed: excluding the only session
    # present left zero sessions to pool, and that only became visible
    # after a real SLURM job had already started and failed.
    if len(session_ids) == 0:
        raise ValueError(
            f"Excluding {sorted(excluded)} leaves ZERO sessions to pool from "
            f"({len(all_session_ids)} found total under {args.mua_large_root}, all "
            f"excluded) -- leave-one-session-out pretraining needs at least one OTHER "
            f"session's data to exist. If you expect more sessions to be available, "
            f"confirm they've actually been built at this dataset root first (e.g. via "
            f"make_huge_dataset.py for each one) before running this again.")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name in ("train", "test"):
        pool_split(args.mua_large_root, session_ids, split_name, args.output_dir)

    print(f"\nDone. Pooled dataset at: {args.output_dir}")
    if excluded:
        print(f"(leave-one-session-out pool -- {sorted(excluded)} excluded entirely, "
              f"safe to fine-tune and evaluate on {'that session' if len(excluded) == 1 else 'those sessions'} "
              f"afterward without any leakage from this pretraining phase)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mua-large-root", type=str, required=True,
                         help="Directory containing one subdirectory per session "
                              "(e.g. datasets/bmi/mua_large), each with its own train/test/*.pkl.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--exclude-session", type=str, action="append", default=None,
                         help="Session id(s) to exclude entirely from the pool (repeatable). "
                              "Use this for the session you intend to fine-tune and evaluate "
                              "on afterward, to avoid any leakage into pretraining's own "
                              "checkpoint selection.")
    args = parser.parse_args()
    main(args)
