"""
Builds a pooled, multi-session dataset directory for pretraining, by
SYMLINKING (not copying -- no data duplication) every session's
train/test .pkl files from an existing mua_8_group-style dataset root
into one unified directory, renamed to avoid filename collisions
(every session's own train/ folder independently starts numbering at
0.pkl, 1.pkl, ...).

Built for a LEAVE-ONE-SESSION-OUT (LOSO) workflow: for each target
session S, pretrain on a pool of every OTHER session (--exclude-session S),
then fine-tune and evaluate on S alone. S's data is never touched --
not for gradients, and not for the pretraining phase's own best_loss
checkpoint selection -- until its own fine-tuning/evaluation.

Safety properties (each was a real or latent leakage path before):
  * The pool is built into a fresh temporary directory and then swapped
    into place. Previously, re-running into an existing output dir with a
    DIFFERENT exclusion left the old links behind -- i.e. the "excluded"
    session's files could silently survive in the pool.
  * After building, every link is verified to resolve into an included
    session's own split folder, and to not be dangling. The check uses the
    link TARGET's path, not the link's filename prefix, so session ids that
    are prefixes of each other (e.g. "s1" vs "s10") cannot confuse it.
  * A pool_manifest.json is written alongside train/ and test/, recording
    exactly which sessions were included/excluded. The LOSO fine-tuning
    script reads it back to confirm its target session was excluded.

CLI usage:
    # List sessions in the canonical order the sbatch arrays index into:
    python build_pretraining_pool.py --mua-large-root ROOT --list-sessions

    # Leave-one-session-out pool for one target session:
    python build_pretraining_pool.py \
        --mua-large-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/indy/mua_8_group \
        --output-dir     /users/bfalkenb/scratch/bfalkenb/data/snn_datasets/bmi/indy/loso_pools/<SESSION> \
        --exclude-session <SESSION>

    # Pool everything (no exclusion -- leaks into checkpoint selection for
    # any session later fine-tuned; kept for comparison only):
    python build_pretraining_pool.py --mua-large-root ROOT --output-dir OUT
"""

import argparse
import datetime
import json
import os
import shutil
import sys

SPLITS = ("train", "test")
MANIFEST_NAME = "pool_manifest.json"


def list_sessions(mua_large_root):
    """Canonical session ordering, shared by every LOSO array script so that
    array index i means the same session everywhere (bash `sort` can order
    differently from Python depending on locale)."""
    return sorted(
        d for d in os.listdir(mua_large_root)
        if os.path.isdir(os.path.join(mua_large_root, d))
    )


def pool_split(mua_large_root, session_ids, split_name, output_dir):
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    n_linked = 0
    per_session = {}
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
            link_path = os.path.join(split_output_dir, f"{session_id}_{fname}")
            if os.path.lexists(link_path):
                # Can only happen if two session ids + filenames collide
                # after renaming -- never silently overwrite.
                raise RuntimeError(f"Link name collision at {link_path}")
            os.symlink(source_path, link_path)
            n_linked += 1
        per_session[session_id] = len(pkl_files)

    print(f"{split_name}: linked {n_linked} files from {len(per_session)} session(s)")
    return n_linked, per_session


def verify_pool(pool_dir, mua_large_root, included, excluded):
    """Every link must resolve into ROOT/<included session>/<split>/ and exist."""
    root = os.path.abspath(mua_large_root)
    included = set(included)
    for split_name in SPLITS:
        split_dir = os.path.join(pool_dir, split_name)
        for name in os.listdir(split_dir):
            link_path = os.path.join(split_dir, name)
            target = os.path.abspath(os.readlink(link_path))
            rel = os.path.relpath(target, root).split(os.sep)
            if len(rel) != 3 or rel[0] == "..":
                raise RuntimeError(f"{link_path} -> {target} is not ROOT/<session>/<split>/<file>")
            session_id, target_split, _ = rel
            if session_id in excluded:
                raise RuntimeError(f"LEAKAGE: {link_path} points into excluded session {session_id}")
            if session_id not in included:
                raise RuntimeError(f"{link_path} points into unexpected session {session_id}")
            if target_split != split_name:
                raise RuntimeError(f"{link_path} crosses splits ({target_split} -> {split_name})")
            if not os.path.exists(link_path):
                raise RuntimeError(f"Dangling link: {link_path} -> {target}")
    print("Verification passed: no links into excluded sessions, none dangling, no split crossing.")


def main(args):
    if not os.path.isdir(args.mua_large_root):
        raise ValueError(f"--mua-large-root {args.mua_large_root} does not exist")
    all_session_ids = list_sessions(args.mua_large_root)

    if args.list_sessions:
        print("\n".join(all_session_ids))
        return

    if not args.output_dir:
        raise ValueError("--output-dir is required unless --list-sessions is given")
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

    # Fail clearly and immediately rather than building an empty pool that
    # only surfaces later as a num_samples=0 crash inside the DataLoader.
    if len(session_ids) == 0:
        raise ValueError(
            f"Excluding {sorted(excluded)} leaves ZERO sessions to pool from "
            f"({len(all_session_ids)} found total under {args.mua_large_root}). "
            f"Leave-one-session-out pretraining needs at least one OTHER session's data.")

    # Build into a fresh sibling temp dir, verify, then swap into place, so
    # stale links from a previous build (possibly with a different exclusion)
    # can never survive, and a crash mid-build never leaves a half-built pool.
    output_dir = os.path.abspath(args.output_dir.rstrip(os.sep))
    tmp_dir = f"{output_dir}.tmp-{os.getpid()}"
    if os.path.lexists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        counts = {}
        for split_name in SPLITS:
            n_linked, per_session = pool_split(args.mua_large_root, session_ids, split_name, tmp_dir)
            if n_linked == 0:
                raise ValueError(f"Pool has zero {split_name} files -- refusing to build an empty split.")
            counts[split_name] = per_session

        verify_pool(tmp_dir, args.mua_large_root, session_ids, excluded)

        manifest = {
            "mua_large_root": os.path.abspath(args.mua_large_root),
            "all_sessions": all_session_ids,
            "included_sessions": session_ids,
            "excluded_sessions": sorted(excluded),
            "files_per_session": counts,
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(tmp_dir, MANIFEST_NAME), "w") as f:
            json.dump(manifest, f, indent=2)

        if os.path.lexists(output_dir):
            shutil.rmtree(output_dir)  # only removes symlinks + manifest, never source data
        os.rename(tmp_dir, output_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"\nDone. Pooled dataset at: {output_dir}")
    if excluded:
        print(f"(leave-one-session-out pool -- {sorted(excluded)} excluded entirely, "
              f"safe to fine-tune and evaluate on "
              f"{'that session' if len(excluded) == 1 else 'those sessions'} afterward)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mua-large-root", type=str, required=True,
                        help="Directory containing one subdirectory per session, "
                             "each with its own train/ and test/ *.pkl files.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--exclude-session", type=str, action="append", default=None,
                        help="Session id(s) to exclude entirely from the pool (repeatable).")
    parser.add_argument("--list-sessions", action="store_true",
                        help="Print session ids in canonical (array-index) order and exit.")
    args = parser.parse_args()
    try:
        main(args)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
