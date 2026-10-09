"""
Long-term decoder performance comparison (RMSE / CC), styled after
Fig. 3a-b (boxplots), Fig. 3c-e (significance matrix), and Fig. 4g-h
(over time) of the reference paper.

Layout: 4 rows x 2 columns
    Row 0: Boxplot RMSE                              | Boxplot CC
    Row 1: Pairwise win-rate matrix (RMSE)           | Pairwise win-rate matrix (CC)
    Row 2: RMSE vs. training-set duration (1-10 min)  | CC vs. training-set duration (1-10 min)
    Row 3: RMSE vs. days since implantation            | CC vs. days since implantation

CLI scoping (this revision): --duration-metrics-path and --speck-summary-path
are both opt-in. If not passed, row 2 is an empty placeholder and
speck/snn_pytorch are not considered at all; no default paths are guessed.
--decoders selects which decoders to include (default: kf wf lstm qrnn snn
speck). Any requested decoder with no data is skipped with a message rather
than causing every session to be dropped.

THIS REVISION updates the script for the current pipeline (several real
changes, not just a rename):

  - MLP is dropped entirely -- removed from single_subject_pipeline.py
    project-wide (not needed for the downstream analyses), so no MLP
    results exist to plot. `decoders` and `colors` below no longer
    include it.

  - Session discovery no longer scans a flat CHECKPOINT_DIR of
    best_model_weights_{session}.pth-style filenames -- that never
    matched the real layout (test_all_decoders.py's checkpoints_dir has
    one SUBDIRECTORY per session, e.g. checkpoints/bmi/mua_large_windowed/
    indy_20160407_02/best_model_weights.pth, matching its own
    discover_sessions()). combined_metrics.json's own top-level keys
    (produced by test_all_decoders.py --multi_session
    --combined_metrics_path ...) already ARE the session IDs -- this
    script now just iterates those directly, removing an entire
    redundant, previously-incompatible discovery step.

  - eval_all_decoders.py was renamed to test_all_decoders.py as part of a
    broader rework (it only ever tests already-trained models, never
    builds one) -- references below updated to match.

  - cc_x/cc_y averaging is now GENUINELY correct, not accidentally so.
    Previously, `cc = dec_entry["cc_x"]` alone (no averaging, despite the
    comment above it claiming otherwise) happened to work only because a
    bug in test_all_decoders.py's own pearson_corrcoef() call was writing
    the x/y AVERAGE into the "cc_x" field and leaving "cc_y" always None
    -- so reading "cc_x" alone was, by accident, already reading the
    paper-convention "Average CC". That upstream bug is now fixed (cc_x
    and cc_y are genuinely separate, per-axis values in freshly-produced
    combined_metrics.json files) -- which means this script's old
    cc-reading line would have silently started reporting x-axis-only CC
    mislabeled as "Average CC" the moment the upstream fix shipped, a
    regression this revision closes by actually averaging cc_x and cc_y
    here, matching what the comment always said it should do.

  - Row 2 (training-duration sweep) no longer includes SNN: no
    duration-tagged SNN checkpoints have been trained yet, so there is no
    real data for that row -- plotting an all-NaN SNN line (technically
    harmless, matplotlib just skips NaN points) still leaves a visually
    misleading empty legend entry implying data that doesn't exist. Rows
    0, 1, and 3 are unaffected and still include SNN, since those come
    from the regular (non-duration) evaluation, which does have SNN
    results.

Row 2 is built from a SEPARATE duration-sweep evaluation run (see
test_all_decoders.py's --train_durations sweep, e.g.
results/multi_session_durations/combined_metrics.json): for each decoder
and each training duration, it aggregates that duration's per-session
RMSE/CC across every session that has a result, and plots the
across-session mean +/- 95% CI (t-distribution CI over the per-session
values -- the same construction used for the within-session chunk CIs
elsewhere) against training duration in minutes. This shows how much
training data each decoder actually needs, independent of any particular
session's total recording length -- which is what the old row 2 measured.

Decoders: kf, wf, lstm, qrnn, snn (roughly worst -> best, matching the
paper's red -> orange -> green -> cyan -> purple -> pink hue progression
along the KF, UKF, WF, WCF, SRNN, GRU, LSTM, QRNN axis) -- this ordering
is COSMETIC ONLY (drives color assignment and the boxplot x-axis order),
not used for any computation: the actual best-performing decoder
(reference_decoder, used for the boxplot significance asterisks) is
computed dynamically from the real RMSE values below, not assumed from
this list's order.
"""

import os
import re
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wilcoxon, t as t_dist

# ---------------------------------------------------------------------------
# 0. Scope this figure to ONE (experiment, subject) -- e.g. bmi/indy --
#    NOT aggregated across subjects or experiments. combined_metrics.json
#    itself is already produced per (experiment, subject) by
#    run_test_all_decoders_array.sbatch + combine_session_metrics.py (see
#    that sbatch script's own RESULTS_DIR), so this script just needs to
#    be pointed at the right ONE rather than a project-wide default.
# ---------------------------------------------------------------------------

# Every decoder this script knows how to draw (has a color, label, etc.),
# in the canonical worst -> best display order. --decoders selects a
# subset of these; the plotted order always follows this list, so colors
# and box positions stay consistent across figures regardless of the
# order the user typed them in.
KNOWN_DECODERS = ["kf", "wf", "lstm", "qrnn", "snn", "snn_pytorch", "speck"]
DEFAULT_DECODERS = ["kf", "wf", "lstm", "qrnn", "snn", "speck"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment", type=str, required=True, choices=["bmi", "hkm"])
parser.add_argument("--subject", type=str, required=True,
                     help="e.g. indy, loco, jenkins, nitschke")
parser.add_argument("--combined-metrics-path", type=str, default=None,
                     help="Default: {results_root}/combined_metrics.json, where "
                          "results_root matches run_test_all_decoders_array.sbatch's "
                          "own RESULTS_DIR for this experiment/subject")
parser.add_argument("--duration-metrics-path", type=str, default=None,
                     help="Output of test_all_decoders.py --multi_session "
                          "--train_durations ... OPTIONAL: if not passed, row 2 is "
                          "left as an empty placeholder (no default path is assumed).")
parser.add_argument("--speck-summary-path", type=str, default=None,
                     help="aggregate_speck_results.py's aggregated_summary.json. "
                          "OPTIONAL: if not passed, speck/snn_pytorch are not "
                          "considered at all, even if listed in --decoders.")
parser.add_argument("--decoders", type=str, nargs="+", default=DEFAULT_DECODERS,
                     help=f"Decoders to include, space- or comma-separated "
                          f"(known: {', '.join(KNOWN_DECODERS)}). Default: "
                          f"{' '.join(DEFAULT_DECODERS)}. Any requested decoder with "
                          f"no data is skipped with a message rather than failing.")
parser.add_argument("--output-path", type=str, default=None,
                     help="Default: decoder_comparison_4x2_{experiment}_{subject}.png")
cli_args = parser.parse_args()

# Accept both "--decoders kf wf lstm" and "--decoders kf,wf,lstm".
requested_decoders = []
for token in cli_args.decoders:
    for name in token.split(","):
        name = name.strip().lower()
        if name and name not in requested_decoders:
            requested_decoders.append(name)
_unknown = [d for d in requested_decoders if d not in KNOWN_DECODERS]
if _unknown:
    print(f"[decoders] ignoring unrecognized decoder name(s) {_unknown} -- "
          f"known: {KNOWN_DECODERS}")
requested_decoders = [d for d in requested_decoders if d in KNOWN_DECODERS]

_RESULTS_ROOT = f"/users/bfalkenb/scratch/bfalkenb/data/results/test_all_decoders/{cli_args.experiment}/{cli_args.subject}"

# ---------------------------------------------------------------------------
# 1. Load combined metrics -- session IDs come directly from this file's
#    own top-level keys (see module docstring for why the old separate
#    checkpoint-directory scan was removed).
# ---------------------------------------------------------------------------
COMBINED_METRICS_PATH = cli_args.combined_metrics_path or os.path.join(
    _RESULTS_ROOT, "combined_metrics.json")
# Both of these are opt-in: None means "not provided", not "use a default".
# DURATION_METRICS_PATH, when given, is the output of test_all_decoders.py
# --multi_session --train_durations 1,2,...,10 (nested {session_id:
# {duration_tag: {decoder: metrics}}}), used only for row 2.
DURATION_METRICS_PATH = cli_args.duration_metrics_path   # None -> row 2 empty
SPECK_SUMMARY_PATH = cli_args.speck_summary_path         # None -> no speck/snn_pytorch

with open(COMBINED_METRICS_PATH, "r") as file:
    data_dict = json.load(file)


# ---------------------------------------------------------------------------
# 1b. Merge in speck ("SNN *speck" -- the real, physical Speck2f chip) and
#     snn_pytorch ("SNN *PyTorch"), from aggregate_speck_results.py's own
#     aggregated_summary.json -- ONLY if --speck-summary-path was passed
#     AND the decoder was requested via --decoders.
# ---------------------------------------------------------------------------
def _merge_impl_from_summary(impl_key, decoder_key, summary, data_dict, combined_metrics_path):
    """Merges one impl's per-session rmse/cc_x/cc_y from an
    aggregate_speck_results.py-style summary dict directly INTO data_dict,
    in the SAME {"rmse":..., "cc_x":..., "cc_y":...} shape every other
    decoder's entry already has -- so every downstream line of this script
    (the skip-if-missing check, the rmses/ccs extraction loop, significance
    testing, plotting) handles it with ZERO further special-casing, the
    same way it already handles kf/wf/lstm/qrnn/snn. Shared by both the
    "speck" and "torch" (as "snn_pytorch") merges below -- same file,
    same shape, same caveats, so one function rather than two copies that
    could quietly drift apart. Returns True only if real data was actually
    merged in (n_merged > 0).
    """
    impl = summary.get("impls", {}).get(impl_key)
    if impl is None:
        print(f"[{impl_key} merge skipped] no '{impl_key}' impl entry in this summary file")
        return False
    sessions = impl["sessions"]
    rmse_vals = impl["rmse_vs_gt"]["values"]
    cc_x_vals = impl["cc_x_vs_gt"]["values"]
    cc_y_vals = impl["cc_y_vs_gt"]["values"]
    n_merged = 0
    for session_id, rmse, cc_x, cc_y in zip(sessions, rmse_vals, cc_x_vals, cc_y_vals):
        if session_id in data_dict and isinstance(data_dict[session_id], dict):
            data_dict[session_id][decoder_key] = {"rmse": rmse, "cc_x": cc_x, "cc_y": cc_y}
            n_merged += 1
    print(f"[{impl_key} merge] {n_merged}/{len(sessions)} {impl_key} session(s) matched a "
          f"session already in {combined_metrics_path} -- see the skip-if-missing check "
          f"below for what happens to any session {impl_key} is missing from.")
    return n_merged > 0


# decoder key in this script -> impl key inside aggregated_summary.json.
# "snn_pytorch" is torch's own eval of the same checkpoint, run by
# infer_snn_speck.py alongside speck -- merged in to let it sit next to
# test_all_decoders.py's own "snn" eval of the identical checkpoint (a
# real, observed discrepancy between the two is exactly what this
# comparison is meant to surface). It is merged only if explicitly
# requested via --decoders, since it's not in the default list.
SUMMARY_IMPLS = {"speck": "speck", "snn_pytorch": "torch"}
requested_summary_decoders = [d for d in requested_decoders if d in SUMMARY_IMPLS]
summary_merged = set()

if SPECK_SUMMARY_PATH is None:
    if requested_summary_decoders:
        print(f"[speck/snn_pytorch skipped] --speck-summary-path not passed -- "
              f"{requested_summary_decoders} will not be considered.")
elif not requested_summary_decoders:
    print(f"[speck/snn_pytorch skipped] --speck-summary-path passed, but neither "
          f"'speck' nor 'snn_pytorch' is in --decoders.")
elif not os.path.isfile(SPECK_SUMMARY_PATH):
    print(f"[speck/snn_pytorch skipped] {SPECK_SUMMARY_PATH} not found -- run "
          f"aggregate_speck_results.py first. The comparison is still produced "
          f"without {requested_summary_decoders}.")
else:
    with open(SPECK_SUMMARY_PATH, "r") as file:
        speck_summary = json.load(file)
    for dec in requested_summary_decoders:
        if _merge_impl_from_summary(SUMMARY_IMPLS[dec], dec, speck_summary,
                                    data_dict, COMBINED_METRICS_PATH):
            summary_merged.add(dec)


def _decoder_available(dec):
    """A decoder is kept only if it has real data in at least one valid
    session. Without this, a requested-but-absent decoder would make every
    session fail the "all decoders present" check below. Summary-sourced
    decoders count only if their own merge actually succeeded, so a stray
    "speck" key in combined_metrics.json can't sneak in without
    --speck-summary-path."""
    if dec in SUMMARY_IMPLS:
        return dec in summary_merged
    return any(isinstance(e, dict) and "error" not in e and dec in e
               for e in data_dict.values())


# Canonical order (KNOWN_DECODERS), not the order typed on the command line
# -- see module docstring: this order is cosmetic only. "snn_pytorch" and
# "speck" sit immediately after "snn" (the decoder they're both variants/
# re-runs of), reading left-to-right: SNN, SNN *PyTorch, SNN *speck.
decoders = [d for d in KNOWN_DECODERS
            if d in requested_decoders and _decoder_available(d)]
_skipped = [d for d in requested_decoders if d not in decoders]
if _skipped:
    print(f"[decoders] skipping {_skipped} -- no results found for them.")
if not decoders:
    raise RuntimeError(
        f"None of the requested decoders {requested_decoders} have results "
        f"in {COMBINED_METRICS_PATH} (or the speck summary, if passed).")
print(f"Decoders included: {decoders}")

# Row 2 never includes SNN variants: there are no duration-tagged SNN
# checkpoints, and speck/snn_pytorch have no duration-sweep concept at all
# (a checkpoint is deployed to the chip as one fixed architecture, not
# trained for a swept number of minutes). Rows 0/1/3 use `decoders`.
duration_decoders = [d for d in decoders if d not in ("snn", "snn_pytorch", "speck")]

# Color scheme chosen to echo the reference figures' hue progression
# (red -> orange -> green -> cyan -> purple -> pink), which in the paper
# roughly tracks worst -> best performing decoder.
colors = {
    "kf": "#e8271c",     # red
    "wf": "#8fdc3c",     # light/chartreuse green
    "snn": "#00bcd4",    # cyan
    "lstm": "#6a1fc9",   # purple/indigo
    "qrnn": "#e91e8c",   # magenta/pink
    "speck": "#00bcd4",       # SAME cyan as "snn" -- both are the SAME
    "snn_pytorch": "#00bcd4",  # underlying decoder, re-run/deployed
    # differently, not separate decoder families; distinguished from
    # each other and from plain "snn" visually by hatching instead (see
    # `hatches`), not by hue.
}

# Hatch patterns distinguish "snn_pytorch"/"speck" from plain "snn" (same
# color for all three) and from EACH OTHER -- see draw_boxplot()'s own use
# of this.
hatches = {"speck": "///", "snn_pytorch": "..."}


def _decoder_display_label(decoder):
    """"speck"/"snn_pytorch" get their own display labels ("SNN *speck"/
    "SNN *PyTorch") rather than the default .upper() -- read as variants
    of SNN, not unrelated decoders of their own. Shared by draw_boxplot()
    and draw_pairwise_matrix() so the two axes never drift apart on
    this."""
    if decoder == "speck":
        return "SNN *speck"
    if decoder == "snn_pytorch":
        return "SNN *PyTorch"
    return decoder.upper()


def stars_for(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    return ""


def _mean_ci(values, confidence=0.95):
    """(mean, ci_low, ci_high) via a t-distribution CI -- same construction
    used for the within-session chunk CIs in test_all_decoders.py, just
    applied here across sessions instead of across chunks."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n > 1:
        sem = values.std(ddof=1) / np.sqrt(n)
        tcrit = t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1)
        margin = float(tcrit * sem)
    else:
        margin = 0.0
    return mean, mean - margin, mean + margin


def _average_cc(dec_entry):
    """Average cc_x and cc_y when both are available (paper reports
    "Average CC"); fall back to cc_x alone if cc_y is absent or None
    (e.g. a combined_metrics.json produced before test_all_decoders.py's
    pearson_corrcoef() call-site fix, where cc_y was always written as
    None -- see module docstring)."""
    cc_x = dec_entry["cc_x"]
    cc_y = dec_entry.get("cc_y")
    if cc_y is None:
        return cc_x
    return (cc_x + cc_y) / 2


# ---------------------------------------------------------------------------
# 2. Build per-decoder RMSE / CC lists, plus date per session
#    (feeds rows 0, 1, and 3 -- the single, full-training-duration eval)
# ---------------------------------------------------------------------------
rmses = {key: [] for key in decoders}
ccs = {key: [] for key in decoders}
# Per-session mean +/- 95% CI (from the n=10-chunk CI now computed and
# stored by test_all_decoders.py). Falls back to no error bar (0-width)
# for older combined_metrics.json entries that predate this field.
rmse_ci_lo = {key: [] for key in decoders}
rmse_ci_hi = {key: [] for key in decoders}
cc_ci_lo = {key: [] for key in decoders}
cc_ci_hi = {key: [] for key in decoders}
dates = []
kept_files = []

for file in sorted(data_dict.keys()):
    entry = data_dict[file]
    if not isinstance(entry, dict) or "error" in entry:
        continue

    # Skip this session entirely if any decoder's results are missing,
    # so all decoder lists stay paired (required for the Wilcoxon test
    # and for consistent date arrays).
    if not all(dec in entry for dec in decoders):
        missing = [dec for dec in decoders if dec not in entry]
        print(f"[skip] {file}: missing decoder results for {missing}")
        continue

    date_match = re.search(r"(\d{8})", file)   # first YYYYMMDD in the session ID (bmi and NWB naming)
    if date_match is None:
        print(f"[skip] {file}: no YYYYMMDD date in the session ID")
        continue
    session_timestamp = datetime.strptime(date_match.group(1), "%Y%m%d").timestamp()

    for decoder in decoders:
        dec_entry = entry[decoder]
        rmse = dec_entry["rmse"]
        cc = _average_cc(dec_entry)
        rmses[decoder].append(rmse)
        ccs[decoder].append(cc)

        # Within-session 95% CI, written by test_all_decoders.py's
        # chunked_rmse_ci / chunked_cc_ci (n=10 chunks per session, matching
        # the reference paper). If a session predates this field, fall back
        # to a zero-width interval centered on the point estimate so the
        # error bar just doesn't show for that session.
        rmse_ci_lo[decoder].append(dec_entry.get("rmse_ci_low", rmse))
        rmse_ci_hi[decoder].append(dec_entry.get("rmse_ci_high", rmse))
        cc_ci_lo[decoder].append(dec_entry.get("cc_ci_low", cc))
        cc_ci_hi[decoder].append(dec_entry.get("cc_ci_high", cc))

    dates.append(session_timestamp)
    kept_files.append(file)

if not kept_files:
    raise RuntimeError("No sessions with complete decoder results were found.")

min_date = min(dates)
days_since_implant = [(d - min_date) / 86400 for d in dates]
days_since_implant = np.array(days_since_implant)

# ---------------------------------------------------------------------------
# 2b. Build training-duration sweep aggregates (row 2) -- duration_decoders
#     only (excludes SNN variants, see module docstring).
#
# DURATION_METRICS_PATH is nested {session_id: {duration_tag: {decoder:
# metrics}}}. For every (decoder, duration) pair, collect the per-session
# RMSE/CC values from every session that has a valid entry there, then
# reduce each pair down to a single across-session (mean, ci_low, ci_high)
# point. Sessions/decoders/durations missing from the file are simply
# excluded from that pair's sample rather than failing the whole run.
#
# GRACEFUL DEGRADATION: if --duration-metrics-path wasn't passed, the file
# doesn't exist, or no selected decoder is eligible for this row, row 2 is
# left empty with an explicit placeholder message -- rows 0/1/3 don't
# depend on this sweep at all and are worth producing regardless.
# ---------------------------------------------------------------------------
all_train_durations = np.array([])
train_dur_rmse_mean = train_dur_rmse_lo = train_dur_rmse_hi = {}
train_dur_cc_mean = train_dur_cc_lo = train_dur_cc_hi = {}
row2_placeholder = "Training-duration sweep\nnot yet run for this config"

if DURATION_METRICS_PATH is None:
    duration_data_available = False
    row2_placeholder = "No training-duration metrics provided\n(--duration-metrics-path)"
    print("[row 2 skipped] --duration-metrics-path not passed -- row 2 will be "
          "an empty placeholder. Rows 0/1/3 are unaffected.")
elif not duration_decoders:
    duration_data_available = False
    row2_placeholder = "No selected decoders have\ntraining-duration results"
    print(f"[row 2 skipped] none of the selected decoders {decoders} are eligible "
          f"for the training-duration row.")
elif not os.path.isfile(DURATION_METRICS_PATH):
    duration_data_available = False
    print(f"[row 2 skipped] {DURATION_METRICS_PATH} not found -- the --train_durations "
          f"sweep hasn't been run for the current config yet. Rows 0/1/3 don't need it "
          f"and will still be produced; row 2 will show a placeholder instead of data.")
else:
    duration_data_available = True
    with open(DURATION_METRICS_PATH, "r") as file:
        duration_data_dict = json.load(file)

    # decoder -> {duration_minutes: [rmse_session_1, rmse_session_2, ...]}
    duration_rmse_samples = {dec: {} for dec in duration_decoders}
    duration_cc_samples = {dec: {} for dec in duration_decoders}

    for session_id, session_entry in duration_data_dict.items():
        if not isinstance(session_entry, dict) or "error" in session_entry:
            continue
        for duration_tag, decoder_entry in session_entry.items():
            # Tags look like "1min", "2.5min", ... (eval_dl_decoders.py's
            # f"{minutes:g}min" convention). Skip anything else (e.g. a stray
            # non-duration key) defensively.
            if not isinstance(decoder_entry, dict) or not duration_tag.endswith("min"):
                continue
            try:
                duration_minutes = float(duration_tag[:-len("min")])
            except ValueError:
                continue
            for decoder in duration_decoders:
                dec_metrics = decoder_entry.get(decoder)
                if dec_metrics is None:
                    continue
                duration_rmse_samples[decoder].setdefault(duration_minutes, []).append(dec_metrics["rmse"])
                duration_cc_samples[decoder].setdefault(duration_minutes, []).append(_average_cc(dec_metrics))

    all_train_durations = sorted({
        d for dec in duration_decoders
        for d in (*duration_rmse_samples[dec].keys(), *duration_cc_samples[dec].keys())
    })
    if not all_train_durations:
        print(f"[row 2 skipped] {DURATION_METRICS_PATH} exists but contains no "
              f"training-duration results for {duration_decoders} -- showing a "
              f"placeholder for row 2 instead of crashing.")
        duration_data_available = False
        row2_placeholder = "Duration metrics file has no\nresults for the selected decoders"
        all_train_durations = np.array([])
    else:
        train_dur_rmse_mean = {dec: [] for dec in duration_decoders}
        train_dur_rmse_lo = {dec: [] for dec in duration_decoders}
        train_dur_rmse_hi = {dec: [] for dec in duration_decoders}
        train_dur_cc_mean = {dec: [] for dec in duration_decoders}
        train_dur_cc_lo = {dec: [] for dec in duration_decoders}
        train_dur_cc_hi = {dec: [] for dec in duration_decoders}

        for d in all_train_durations:
            for dec in duration_decoders:
                rmse_vals = duration_rmse_samples[dec].get(d)
                if rmse_vals:
                    mean, lo, hi = _mean_ci(rmse_vals)
                else:
                    mean = lo = hi = np.nan
                train_dur_rmse_mean[dec].append(mean)
                train_dur_rmse_lo[dec].append(lo)
                train_dur_rmse_hi[dec].append(hi)

                cc_vals = duration_cc_samples[dec].get(d)
                if cc_vals:
                    mean, lo, hi = _mean_ci(cc_vals)
                else:
                    mean = lo = hi = np.nan
                train_dur_cc_mean[dec].append(mean)
                train_dur_cc_lo[dec].append(lo)
                train_dur_cc_hi[dec].append(hi)

        all_train_durations = np.array(all_train_durations)

all_train_durations = np.array(all_train_durations)

# ---------------------------------------------------------------------------
# 3. Significance testing (two-tailed paired Wilcoxon signed-rank test,
#    Holm-corrected within each panel -- same procedure as report_figures.py)
# ---------------------------------------------------------------------------
def _wilcoxon_p(a, b):
    try:
        return wilcoxon(a, b)[1]
    except ValueError:      # e.g. all paired differences are zero
        return 1.0


def _holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


# 3a. Each decoder vs. the best-performing decoder (lowest mean RMSE),
#     used for the asterisks above the boxplots (panels a/b), mirroring
#     the paper's comparison against QRNN. Holm-corrected over the
#     comparisons within each panel.
reference_decoder = min(decoders, key=lambda d: np.mean(rmses[d]))
print(f"Reference decoder for significance testing: {reference_decoder}")


def significance_labels(metric_dict, reference):
    others = [d for d in decoders if d != reference]
    p_adj = _holm([_wilcoxon_p(metric_dict[d], metric_dict[reference]) for d in others])
    labels = {reference: ""}
    labels.update({d: stars_for(p) for d, p in zip(others, p_adj)})
    return labels


rmse_sig = significance_labels(rmses, reference_decoder)
cc_sig = significance_labels(ccs, reference_decoder)
print(f"Sessions in the paired tests: {len(kept_files)}"
      + ("  (WARNING: with n<6 a two-sided Wilcoxon test cannot reach p<0.05 -- "
         "read the win-rate colors, not the stars)" if len(kept_files) < 6 else ""))

# ---------------------------------------------------------------------------
# 4. Plotting helpers
# ---------------------------------------------------------------------------
mean_marker = dict(marker="o", markerfacecolor="white", markeredgecolor="black",
                    markersize=6, linewidth=1.2)


def draw_boxplot(ax, metric_dict, sig_labels, ylabel):
    tick_label_text = [_decoder_display_label(d) for d in decoders]
    try:
        # matplotlib >= 3.9 renamed 'labels' to 'tick_labels' -- see
        # test_all_decoders.py's identical fallback for this same issue.
        box = ax.boxplot(
            [metric_dict[d] for d in decoders],
            tick_labels=tick_label_text,
            patch_artist=True,
            showmeans=True,
            meanprops=mean_marker,
            widths=0.6,
        )
    except TypeError:
        box = ax.boxplot(
            [metric_dict[d] for d in decoders],
            labels=tick_label_text,
            patch_artist=True,
            showmeans=True,
            meanprops=mean_marker,
            widths=0.6,
        )

    # Rotated -- with 7 decoders possible ("SNN *PyTorch"/"SNN *speck"
    # both genuinely long labels), horizontal tick labels overlap.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for patch, decoder in zip(box["boxes"], decoders):
        patch.set_facecolor(colors[decoder])
        patch.set_edgecolor("black")
        if decoder in hatches:
            patch.set_hatch(hatches[decoder])

    for i, decoder in enumerate(decoders):
        for whisker in box["whiskers"][2 * i: 2 * i + 2]:
            whisker.set_color("black")
        for cap in box["caps"][2 * i: 2 * i + 2]:
            cap.set_color("black")

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Significance asterisks above each box
    ymax = max(max(v) for v in metric_dict.values())
    ypad = (ymax - min(min(v) for v in metric_dict.values())) * 0.05
    for i, decoder in enumerate(decoders, start=1):
        label = sig_labels[decoder]
        if label:
            top = max(metric_dict[decoder])
            ax.text(i, top + ypad, label, ha="center", va="bottom", fontsize=11)

    ax.set_ylabel(ylabel)


def draw_pairwise_matrix(ax, metric_dict, higher_is_better):
    """Cell (row, col): how often and by how much the row decoder beats the
    column decoder over the paired sessions. Colour = percentage of sessions
    in which the row decoder is better (ties count half; 50% = no consistent
    winner); text = median paired difference row - col in the metric's units,
    with stars from the two-sided Wilcoxon signed-rank test, Holm-corrected
    over all pairs ('n.s.' otherwise). The two triangles mirror each other
    (100 - % and -diff)."""
    n = len(decoders)
    sign = 1 if higher_is_better else -1
    win = np.full((n, n), np.nan)
    diff = np.full((n, n), np.nan)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    p_adj = _holm([_wilcoxon_p(metric_dict[decoders[i]], metric_dict[decoders[j]])
                   for i, j in pairs])
    pmat = np.full((n, n), np.nan)
    for (i, j), p in zip(pairs, p_adj):
        pmat[i, j] = pmat[j, i] = p
    for i, a in enumerate(decoders):
        for j, b in enumerate(decoders):
            if i != j:
                d = np.asarray(metric_dict[a], dtype=float) - np.asarray(metric_dict[b], dtype=float)
                diff[i, j] = np.median(d)
                win[i, j] = 100 * (np.mean(sign * d > 0) + 0.5 * np.mean(d == 0))

    # Keep cells a sensible size however few decoders there are: draw into a centred inset
    # sized for n/6 of the panel width (never wider than the panel), so a 3-decoder matrix has
    # the same cell size as a 6-decoder one instead of three huge blocks.
    frac = min(1.0, max(0.5, n / 6.0))
    host = ax
    host.axis("off")
    ax = host.inset_axes([(1 - frac) / 2, 0.0, frac, 1.0])

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="0.85")   # grey diagonal
    im = ax.imshow(np.ma.masked_invalid(win), cmap=cmap, vmin=0, vmax=100, aspect="equal")

    labels = [_decoder_display_label(d) for d in decoders]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    for i in range(n):
        for j in range(n):
            if i != j:
                stars = stars_for(pmat[i, j])
                text = f"{diff[i, j]:+.2g}" + (f"\n{stars}" if stars else "\nn.s.")
                ax.text(j, i, text, ha="center", va="center", fontsize=7.5, linespacing=1.1,
                        color="white" if abs(win[i, j] - 50) > 35 else "black")

    cax = make_axes_locatable(ax).append_axes("bottom", size="6%", pad=0.55)
    bar = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.set_label("Sessions where the row decoder is better (%)")


def _asymmetric_yerr(y, lo, hi):
    """Build the (2, n) yerr array errorbar() expects from lower/upper
    CI bounds, clipping tiny negative rounding noise to zero."""
    y = np.asarray(y)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    err_low = np.clip(y - lo, 0, None)
    err_high = np.clip(hi - y, 0, None)
    return np.vstack([err_low, err_high])


def draw_over_categorical(
    ax,
    x_values,
    metric_dict,
    ci_lo_dict,
    ci_hi_dict,
    ylabel,
    xlabel,
    tick_formatter=str,
    clip_decoder=None,
    clip_margin=1.15,
    decoders_to_plot=None,
):
    """
    Plot decoder performance over an ordered categorical variable.

    The x-values are sorted and displayed with equal spacing while retaining
    their original values as tick labels. NaN entries in a decoder's series
    (e.g. no cached model for that decoder at that duration) leave a gap in
    that decoder's line rather than raising.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.

    x_values : array-like
        Values used to order the points (e.g., training duration or
        implantation day).

    metric_dict : dict
        Dictionary mapping decoder -> metric values, parallel to x_values.

    ci_lo_dict : dict
        Dictionary mapping decoder -> lower confidence interval.

    ci_hi_dict : dict
        Dictionary mapping decoder -> upper confidence interval.

    ylabel : str
        Y-axis label.

    xlabel : str
        X-axis label.

    tick_formatter : callable, optional
        Function converting each x-value into a tick label.

    clip_decoder : str, optional
        If set, this decoder is EXCLUDED when computing the y-axis limit,
        and any of its points that fall above the resulting limit are drawn
        as an upward arrow + value label at the top of the axis instead of
        stretching the whole plot to fit them. The point is never silently
        dropped -- it's annotated, not hidden.

    clip_margin : float, optional
        Multiplier applied to the largest non-clip_decoder upper CI bound
        to set the y-axis ceiling (default 1.15, i.e. 15% headroom).

    decoders_to_plot : list, optional
        Which decoders to draw -- defaults to the global `decoders` list.
    """
    plot_decoders = decoders_to_plot if decoders_to_plot is not None else decoders

    # Sort by the supplied x-values
    order = np.argsort(x_values)
    values_sorted = np.asarray(x_values)[order]

    # Evenly spaced x locations
    x_pos = np.arange(len(values_sorted))

    for decoder in plot_decoders:
        y = np.asarray(metric_dict[decoder])[order]
        lo = np.asarray(ci_lo_dict[decoder])[order]
        hi = np.asarray(ci_hi_dict[decoder])[order]

        yerr = _asymmetric_yerr(y, lo, hi)

        ax.errorbar(
            x_pos,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            color=colors[decoder],
            linewidth=1.5,
            markersize=5,
            capsize=2,
            elinewidth=1,
            label=decoder.upper(),
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [tick_formatter(v) for v in values_sorted],
        rotation=90,
        fontsize=8,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Small horizontal margin so the first/last points aren't flush
    ax.margins(x=0.02)

    if clip_decoder is not None and clip_decoder in metric_dict and clip_decoder in plot_decoders:
        other_highs = [
            np.nanmax(np.asarray(ci_hi_dict[d])) for d in plot_decoders if d != clip_decoder
        ]
        other_highs = [h for h in other_highs if np.isfinite(h)]
        if other_highs:
            y_cap = max(other_highs) * clip_margin
            y_clip = np.asarray(metric_dict[clip_decoder])[order]
            ax.set_ylim(0, y_cap)
            for xi, yi in zip(x_pos, y_clip):
                if np.isfinite(yi) and yi > y_cap:
                    ax.annotate(
                        f"{clip_decoder.upper()}\n{yi:.0f}",
                        xy=(xi, y_cap), xytext=(xi, y_cap * 0.90),
                        ha="center", va="top", fontsize=7,
                        color=colors.get(clip_decoder, "black"),
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>",
                                         color=colors.get(clip_decoder, "black"),
                                         lw=1.2),
                    )


# ---------------------------------------------------------------------------
# 5. Assemble the 4x2 figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(12, 20))

# Row 0 (a, b): boxplots
draw_boxplot(axes[0, 0], rmses, rmse_sig, "Average RMSE")
draw_boxplot(axes[0, 1], ccs, cc_sig, "Average CC")

# Row 1 (c, d): pairwise win-rate matrices (lower RMSE / higher CC wins)
draw_pairwise_matrix(axes[1, 0], rmses, higher_is_better=False)
draw_pairwise_matrix(axes[1, 1], ccs, higher_is_better=True)

# Row 2: RMSE/CC vs. TRAINING duration (across-session mean +/- 95% CI,
# from the --train_durations sweep) -- duration_decoders only.
#
# WF gets clip_decoder treatment on the RMSE panel ONLY: its unregularized
# fit blows up at the 2-minute mark (a diagnosed, known implementation
# issue), and left unclipped it stretches this panel's y-axis to ~700+.
# The CC panel doesn't need this -- CC is bounded to [-1, 1].
if duration_data_available:
    draw_over_categorical(
        axes[2, 0],
        all_train_durations,
        train_dur_rmse_mean,
        train_dur_rmse_lo,
        train_dur_rmse_hi,
        ylabel="RMSE",
        xlabel="Training Duration (min)",
        tick_formatter=lambda x: f"{x:g}",
        clip_decoder="wf",
        decoders_to_plot=duration_decoders,
    )
    draw_over_categorical(
        axes[2, 1],
        all_train_durations,
        train_dur_cc_mean,
        train_dur_cc_lo,
        train_dur_cc_hi,
        ylabel="Correlation",
        xlabel="Training Duration (min)",
        tick_formatter=lambda x: f"{x:g}",
        decoders_to_plot=duration_decoders,
    )
else:
    for ax in (axes[2, 0], axes[2, 1]):
        ax.text(0.5, 0.5, row2_placeholder, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")

# Row 3 (g, h): RMSE/CC vs. days since implantation -- full decoder list,
# including SNN (this comes from the regular, non-duration evaluation) --
# EXCLUDING speck/snn_pytorch, deliberately: speck is scoped to rows 0/1
# only, and snn_pytorch follows the same scoping for consistency.
row3_decoders = [d for d in decoders if d not in ("speck", "snn_pytorch")]
draw_over_categorical(
    axes[3, 0],
    days_since_implant,
    rmses,
    rmse_ci_lo,
    rmse_ci_hi,
    ylabel="RMSE",
    xlabel="Days Since Implantation",
    tick_formatter=lambda x: str(int(round(x))),
    decoders_to_plot=row3_decoders,
)

draw_over_categorical(
    axes[3, 1],
    days_since_implant,
    ccs,
    cc_ci_lo,
    cc_ci_hi,
    ylabel="Correlation",
    xlabel="Days Since Implantation",
    tick_formatter=lambda x: str(int(round(x))),
    decoders_to_plot=row3_decoders,
)

# Shared legend from row 3 (the full non-speck decoder list, including SNN)
# rather than row 2 (which deliberately excludes SNN).
handles, labels = axes[3, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.01), frameon=False)

for ax, letter in zip(axes.flat, "abcdefgh"):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

plt.tight_layout(rect=[0, 0, 1, 0.97])
_output_path = cli_args.output_path or f"decoder_comparison_4x2_{cli_args.experiment}_{cli_args.subject}.png"
plt.savefig(_output_path, dpi=200, bbox_inches="tight")
print(f"Saved {_output_path}")
plt.show()
