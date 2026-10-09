"""
Aggregate version of plot_decoder_efficiency.py's figure, across every
session that has both a per-session profile (latency + param_count, from
plot_decoder_efficiency.py --profile_save_path) and RMSE results (from
combined_metrics.json). Same figure, same caveats (see that script's
module docstring), but now: x-position, y-position, and marker size are
all across-session MEANS, and x/y both get error bars (95% CI, same
t-distribution construction used throughout this project's other
aggregate figures -- decoder_comparison_4x2.py's _mean_ci()) rather than
a single session's point estimate.

Why aggregating each of the three axes is (or isn't) actually meaningful,
worth being explicit about rather than aggregating everything uniformly
just for consistency:

  - LATENCY genuinely should vary session to session, even for the exact
    same model: it depends on real system conditions at measurement time
    (CPU load, which node a job happened to land on, thermal state).
    Averaging across many sessions' measurements is a real improvement
    over one session's noisy point estimate, not just cosmetic.
  - PARAM COUNT is architecture-determined, not data-determined -- with
    MUA fixed at 96 channels across every session (process_data.py's own
    hardcoded num_chan), every decoder's parameter count SHOULD be
    identical across sessions. Averaging it is harmless, but the more
    useful thing this script does with it is check that it's actually
    constant and print a warning if it isn't -- that would be a genuinely
    surprising, worth-investigating finding, not something to silently
    average over.
  - RMSE was already aggregated (as a bare mean) by
    plot_decoder_efficiency.py's own compute_mean_rmse() -- this script
    recomputes it directly from combined_metrics.json instead, purely to
    also get a CI, matching every other aggregate figure in this project
    rather than being the one exception without error bars.

CLI usage:
    python plot_decoder_efficiency_aggregate.py \
        --profiles_dir results/decoder_efficiency_profiles \
        --combined_metrics_path results/multi_session/combined_metrics.json \
        --decoders lstm,qrnn,kf,wf,snn \
        --save_path results/decoder_efficiency_aggregate.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

from plot_decoder_efficiency import COLORS, HATCHES, make_efficiency_figure


def _mean_ci(values, confidence=0.95):
    """(mean, ci_low, ci_high) via a t-distribution CI -- identical
    construction to decoder_comparison_4x2.py's own _mean_ci(), so error
    bars across this project's aggregate figures mean the same thing."""
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


def load_profiles(profiles_dir):
    """Returns {decoder: {'latency_s': [...], 'param_count': [...]}}
    across every *_profile.json found under profiles_dir (written by
    plot_decoder_efficiency.py --profile_save_path).

    BUG FIX: was a FLAT glob (os.path.join(profiles_dir, "*.json")), which only matched files
    sitting directly inside profiles_dir -- found NOTHING against a real sbatch array job's own
    output, which (matching how every other script in this project organizes results) wrote to
    {profiles_dir}/{experiment}/{subject}/{session}_profile.json, one directory level (at
    least) deeper than the flat glob ever looked. Now RECURSIVE (glob's own `**`, recursive=True),
    so profiles are found at any depth under profiles_dir, matching that convention -- confirmed
    against a real run that failed with "No profile JSON files found" despite the files
    genuinely being there, just nested.

    MIXING SUBJECTS WARNING: profiles_dir is not required to be subject-specific, and a
    recursive glob will happily find and pool sessions from MULTIPLE subjects (e.g. both
    bmi/indy/ and bmi/loco/) into ONE aggregate if profiles_dir is pointed at their shared
    parent -- nothing this project has built elsewhere does that (decoder_comparison_4x2.py
    requires --subject; aggregate_speck_results.py's --results-root is always one subject's own
    directory), so a mixed-subject aggregate here would be a first, and likely not what's
    wanted: averaging RMSE/latency across two different subjects' recordings blends two
    genuinely different signals into one number that describes neither. This function can't
    know your INTENT, so it detects the situation from the profile files' own session names
    (matching this project's own {subject}_{date}_{run} convention) and prints a clear,
    impossible-to-miss warning if more than one subject's sessions were found together --
    it does not refuse to proceed, since a genuinely-intended multi-subject aggregate isn't
    this function's call to block, only to flag.
    """
    files = sorted(glob.glob(os.path.join(profiles_dir, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No profile JSON files found under {profiles_dir} (searched recursively) -- has "
            f"the profiling array job (plot_decoder_efficiency.py --profile_save_path) actually "
            f"finished any sessions yet? If you know the files exist, double check "
            f"{profiles_dir} is the right root -- this now searches every subdirectory, so it "
            f"is no longer a flat-vs-nested path problem if it still can't find them.")

    per_decoder = {}
    n_sessions_loaded = 0
    subjects_seen = set()
    energy_methods_seen = {}  # decoder -> {method -> n_sessions} -- for the consistency check below
    for filepath in files:
        with open(filepath, "r") as f:
            content = json.load(f)
        if "decoders" not in content:
            print(f"[skip] {os.path.basename(filepath)}: no 'decoders' key -- unexpected "
                  f"file structure, not plot_decoder_efficiency.py's own profile format")
            continue
        # {subject}_{date}_{run} -- this project's session-naming convention throughout
        # (e.g. 'indy_20160407_02') -- used ONLY for the mixed-subject warning below, not for
        # anything that affects what gets loaded or how it's aggregated.
        session_name = content.get("session") or os.path.splitext(os.path.basename(filepath))[0]
        subject_guess = session_name.split("_")[0] if "_" in session_name else session_name
        subjects_seen.add(subject_guess)
        # energy_method is stored ONCE per FILE (a property of the node this session's job
        # landed on -- whether RAPL was readable there -- not of any individual decoder; see
        # plot_decoder_efficiency.py's own comment on this same field). Read once per file,
        # applied to every decoder in it below.
        file_energy_method = content.get("energy_method")
        n_sessions_loaded += 1
        for name, metrics in content["decoders"].items():
            entry = per_decoder.setdefault(name, {"latency_s": [], "param_count": [], "energy_j": []})
            entry["latency_s"].append(metrics["latency_s"])
            entry["param_count"].append(metrics["param_count"])
            # energy_j is None whenever --skip_energy was set for that run, OR (rarer) a single
            # session's own RAPL counter wrapped mid-measurement (see energy_meter.py) -- kept
            # as None here too, not silently dropped or zeroed, so build_aggregate_records() can
            # report exactly how many of N sessions actually have an energy figure rather than
            # quietly averaging over fewer sessions than the RMSE/latency columns show.
            entry["energy_j"].append(metrics.get("energy_j"))
            if file_energy_method is not None:
                energy_methods_seen.setdefault(name, {}).setdefault(file_energy_method, 0)
                energy_methods_seen[name][file_energy_method] += 1

    print(f"Loaded {n_sessions_loaded} session profile(s) from {profiles_dir}")
    if len(subjects_seen) > 1:
        print(f"\n{'!' * 72}")
        print(f"  WARNING: profiles from {len(subjects_seen)} DIFFERENT subjects were found and "
              f"will be POOLED together into one aggregate: {sorted(subjects_seen)}. Nothing "
              f"else in this project mixes subjects this way -- if this isn't intended, point "
              f"--profiles_dir at one subject's own subdirectory instead (e.g. "
              f"'{profiles_dir}/bmi/indy' rather than '{profiles_dir}').")
        print(f"{'!' * 72}\n")
    # energy_method can genuinely DIFFER session to session on a shared cluster: a SLURM array
    # job's tasks land on different nodes, and if some expose readable RAPL counters while
    # others don't (plausible on heterogeneous cluster hardware), some sessions' energy_j would
    # be a real measurement and others a rough psutil-based proxy -- averaged together, that's
    # not one honest number. Flagged here rather than silently blended; build_aggregate_records()
    # reports whichever method is a MAJORITY, but this is the place that would tell you if that
    # majority isn't unanimous.
    for name, methods in energy_methods_seen.items():
        if len(methods) > 1:
            print(f"  WARNING: {name.upper()}'s energy_method is NOT consistent across sessions: "
                  f"{methods} -- some sessions measured real RAPL energy, others used the "
                  f"psutil proxy (or vice versa). Averaging these together mixes a measurement "
                  f"with an estimate; investigate which nodes lack RAPL before trusting this "
                  f"decoder's aggregate energy figure.")
    return per_decoder, energy_methods_seen


def load_rmse(combined_metrics_path, decoders):
    """Returns {decoder: [rmse_session_1, rmse_session_2, ...]} across
    every session in combined_metrics_path that has a result for that
    decoder -- same reading pattern as decoder_comparison_4x2.py."""
    with open(combined_metrics_path, "r") as f:
        data = json.load(f)
    rmses = {d: [] for d in decoders}
    for entry in data.values():
        if not isinstance(entry, dict) or "error" in entry:
            continue
        for d in decoders:
            if d in entry:
                rmses[d].append(entry[d]["rmse"])
    return rmses


def merge_impl_from_summary(impl_key, decoder_key, speck_summary_path, per_decoder, rmses,
                            energy_methods_seen):
    """Injects ONE impl (e.g. 'speck', or 'torch' as 'snn_pytorch') from an
    aggregate_speck_results.py-style summary directly into per_decoder/rmses, in the SAME
    shape every other decoder's entry already has -- so build_aggregate_records() handles it
    with ZERO further changes, the same merge-not-special-case approach
    decoder_comparison_4x2.py already uses for the SAME two impls from the SAME summary file
    (its own _merge_impl_from_summary()). Generalized from an earlier speck-only version of
    this function specifically so "torch" (as "snn_pytorch") could be merged the same way --
    infer_snn_speck.py's own real, laptop-measured torch latency/power, run on the same
    checkpoint/sessions as speck, was previously entirely ABSENT from this figure; only
    test_all_decoders.py's OWN, differently-measured "snn" row (see time_snn_per_timestep()'s
    module docstring -- it deliberately times ONE call per 4ms timestep, unlike
    infer_snn_speck.py's one-call-per-trial "torch", a genuine methodology difference, not a
    bug in either) represented the SNN at all.

    Returns True if real data was actually merged in, False otherwise (missing file, or a
    file with no `impl_key` impl entry) -- the caller uses this to decide whether
    `decoder_key` belongs in `decoders` at all, so an experiment/subject without any data for
    THIS impl yet still produces the original figure rather than crashing or silently showing
    an empty/zero point.

    UNIT NOTE: aggregate_speck_results.py's own impls[impl_key]['latency_per_timestep_ms']
    ['values'] is already in MILLISECONDS (one value per session) -- per_decoder[...]
    ['latency_s'] expects SECONDS (see make_efficiency_figure()'s own `* 1000` at the point
    of display), so this divides by 1000 on the way in, once, here -- not at every later
    point that reads per_decoder.

    ENERGY: neither impl's own aggregate stores an energy_j figure directly -- both store
    power_mw (a real measurement: RAPL for torch, PowerMonitor for speck) and
    latency_per_timestep_ms separately. energy_j = power (W) x latency (s) is exactly the
    same quantity every ANN decoder's own energy_j already is (RAPL/proxy also measure a
    power draw over a window and report the resulting energy for that window) -- just
    derived here from each impl's two already-real numbers instead of read directly. Uses
    each SESSION's own paired power/latency values, not the pre-aggregated means, so this is
    per-session energy exactly like every other decoder's energy_j list, and
    build_aggregate_records() can average it the same way. energy_method is read directly
    from the summary (see aggregate_speck_results.py's own energy_method tracking) rather
    than assumed -- torch's is 'rapl' on every real run seen so far, but asserting that
    without checking is exactly the kind of unverified assumption this project keeps finding
    bugs from.
    """
    if not os.path.isfile(speck_summary_path):
        print(f"[{impl_key} merge skipped] {speck_summary_path} not found -- run "
              f"aggregate_speck_results.py first, or pass --speck_summary_path, to add "
              f"the '{decoder_key}' point. The original decoder comparison is still "
              f"produced without it.")
        return False

    with open(speck_summary_path, "r") as f:
        speck_summary = json.load(f)
    impl = speck_summary.get("impls", {}).get(impl_key)
    if impl is None:
        print(f"[{impl_key} merge skipped] {speck_summary_path} has no '{impl_key}' impl entry")
        return False

    latency_ms_values = impl["latency_per_timestep_ms"]["values"]
    rmse_values = impl["rmse_vs_gt"]["values"]
    param_count = speck_summary.get("dynapcnn_param_count")
    if param_count is None:
        print(f"[{impl_key} merge skipped] {speck_summary_path} has no dynapcnn_param_count "
              f"(or sessions disagreed on it -- see aggregate_speck_results.py's own "
              f"dynapcnn_param_count_warning) -- make_efficiency_figure() needs a real "
              f"param_count for marker sizing, so '{decoder_key}' can't be added without one.")
        return False

    power_mw_entry = impl.get("power_mw", {})
    power_mw_values = power_mw_entry.get("values")
    if power_mw_values is not None and len(power_mw_values) == len(latency_ms_values):
        # energy_j = power (W) x latency (s), per session -- see this function's own docstring
        # ("ENERGY:") for why this is the same quantity every ANN decoder's energy_j already is.
        energy_j_values = [(pw / 1000) * (ms / 1000) for pw, ms in zip(power_mw_values, latency_ms_values)]
    else:
        print(f"  [{impl_key} energy] no usable power_mw in {speck_summary_path} -- "
              f"{decoder_key}'s energy will show as unavailable rather than derived from a "
              f"mismatched or missing field.")
        energy_j_values = [None] * len(latency_ms_values)

    per_decoder[decoder_key] = {
        "latency_s": [ms / 1000 for ms in latency_ms_values],
        "param_count": [param_count] * len(latency_ms_values),
        "energy_j": energy_j_values,
    }
    rmses[decoder_key] = rmse_values
    if impl.get("energy_method") is not None:
        energy_methods_seen[decoder_key] = {impl["energy_method"]: len(latency_ms_values)}
    print(f"[{impl_key} merge] {len(latency_ms_values)} session(s) merged as '{decoder_key}' "
          f"from {speck_summary_path}")
    return True


def check_param_count_consistency(per_decoder):
    """Param count should be IDENTICAL across sessions (MUA is fixed at
    96 channels project-wide) -- prints a clear warning, not a silent
    average, if any decoder's count actually varies. See module
    docstring for why this is worth checking rather than assuming."""
    for name, entry in per_decoder.items():
        counts = set(entry["param_count"])
        if len(counts) > 1:
            print(f"WARNING: {name.upper()}'s param_count VARIES across sessions: {sorted(counts)} "
                  f"-- expected a single constant value (MUA is fixed at 96 channels project-"
                  f"wide). This is a genuinely surprising finding worth investigating directly, "
                  f"not something to silently average over.")


def build_aggregate_records(per_decoder, rmses, decoders, energy_methods_seen=None):
    """One record per decoder with enough data to plot, each carrying its
    own CI bounds for the figure's error bars.

    ENERGY is handled separately from latency/RMSE/param_count: it's legitimately allowed to
    be missing (a --skip_energy profiling run, or a single session's RAPL wraparound -- see
    load_profiles()) where the other three fields never are, so a decoder can have a full,
    valid record WITHOUT an energy figure. record['energy_j'] is None, and
    record['n_energy_sessions'] is 0, in that case -- callers must check for None before
    plotting or printing energy, the same discipline this project has applied to every other
    optional field (RAPL wraparound, missing dynapcnn_param_count, etc.)."""
    records = []
    ci_bounds = {}  # name -> (latency_lo, latency_hi, rmse_lo, rmse_hi, energy_lo, energy_hi)
    energy_methods_seen = energy_methods_seen or {}
    for name in decoders:
        if name not in per_decoder:
            print(f"[skip] {name}: no profiling data found")
            continue
        if not rmses.get(name):
            print(f"[skip] {name}: no RMSE data found in combined_metrics_path")
            continue

        latency_mean, latency_lo, latency_hi = _mean_ci(per_decoder[name]["latency_s"])
        rmse_mean, rmse_lo, rmse_hi = _mean_ci(rmses[name])
        param_mean = float(np.mean(per_decoder[name]["param_count"]))

        energy_values = [e for e in per_decoder[name].get("energy_j", []) if e is not None]
        n_energy = len(energy_values)
        if n_energy > 0:
            energy_mean, energy_lo, energy_hi = _mean_ci(energy_values)
        else:
            energy_mean = energy_lo = energy_hi = None

        # speck's energy is always a real chip measurement, by construction (see
        # merge_speck_summary()); every other decoder's is whatever plot_decoder_efficiency.py
        # recorded for that session (real RAPL, or the psutil proxy if RAPL wasn't readable on
        # the node that session ran on -- see load_profiles()'s own cross-session consistency
        # check for when this isn't unanimous).
        # Gated on n_energy > 0: energy_methods_seen can record a method for a decoder
        # (profile-JSON decoders: a file's top-level energy_method, applied to every decoder
        # in it; snn_pytorch/speck: read directly from aggregate_speck_results.py -- see
        # merge_impl_from_summary()) even when that decoder's OWN energy_j came back None for
        # every session (a per-decoder --skip_energy-style gap, or a summary with no power_mw)
        # -- without this gate, a decoder with ZERO usable energy values could still get
        # labeled with a method, describing a number that doesn't exist. Uniform across every
        # decoder including snn_pytorch/speck now that merge_impl_from_summary() populates
        # energy_methods_seen the same way load_profiles() does for the profiled ANN decoders
        # -- no more speck-specific special case needed here.
        methods = energy_methods_seen.get(name, {})
        energy_method = max(methods, key=methods.get) if (n_energy > 0 and methods) else None

        records.append({"name": name, "rmse": rmse_mean, "latency_s": latency_mean,
                         "param_count": param_mean, "energy_j": energy_mean,
                         "energy_lo": energy_lo, "energy_hi": energy_hi,
                         "energy_method": energy_method, "n_energy_sessions": n_energy,
                         "n_sessions": len(per_decoder[name]["latency_s"])})
        ci_bounds[name] = (latency_lo, latency_hi, rmse_lo, rmse_hi, energy_lo, energy_hi)

    return records, ci_bounds


def add_error_bars(ax, records, ci_bounds):
    """Adds x (latency) and y (RMSE) 95% CI error bars to an
    already-drawn make_efficiency_figure() axis -- called AFTER that
    function's own scatter/annotate calls, not integrated into it, so the
    single-session figure (which has no CI to show -- one session is one
    point) is completely unaffected by this addition."""
    for rec in records:
        latency_lo, latency_hi, rmse_lo, rmse_hi, _energy_lo, _energy_hi = ci_bounds[rec["name"]]
        x = rec["latency_s"] * 1000
        y = rec["rmse"]
        xerr = np.array([[max(0, x - latency_lo * 1000)], [max(0, latency_hi * 1000 - x)]])
        yerr = np.array([[max(0, y - rmse_lo)], [max(0, rmse_hi - y)]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none',
                     ecolor=COLORS.get(rec["name"], 'gray'), elinewidth=1.2,
                     capsize=3, zorder=2, alpha=0.7)


# energy_method values that mean "a real hardware measurement" vs. "a software-side guess" --
# the ONLY thing make_energy_bar_figure() uses to decide which of its two panels a decoder
# belongs in. Anything not in _REAL_ENERGY_METHODS (currently just 'proxy_psutil') is treated
# as an estimate; an unrecognized future method name therefore defaults to the estimate panel,
# not the measured one -- the safer direction to be wrong in.
_REAL_ENERGY_METHODS = {"rapl", "chip_power_monitor"}


def _draw_energy_panel(ax, usable, title):
    """One bar per decoder in `usable`, shared by both panels of make_energy_bar_figure() --
    identical bar/error-bar/label drawing either way, just called once per panel with that
    panel's own Axes and its own subset of records, so each panel gets its OWN y-axis scale
    (see make_energy_bar_figure()'s own docstring for why the two must never share one)."""
    energies_uj = [r["energy_j"] * 1e6 for r in usable]  # J -> uJ, readable range for this project
    x = np.arange(len(usable))
    for i, rec in enumerate(usable):
        ax.bar(i, energies_uj[i], color=COLORS.get(rec["name"], "gray"),
               edgecolor="black", linewidth=0.8, hatch=HATCHES.get(rec["name"]), zorder=3)
        if rec.get("n_energy_sessions", 0) > 1 and rec.get("energy_lo") is not None:
            e_lo_uj = rec["energy_lo"] * 1e6
            e_hi_uj = rec["energy_hi"] * 1e6
            yerr = np.array([[max(0, energies_uj[i] - e_lo_uj)], [max(0, e_hi_uj - energies_uj[i])]])
            ax.errorbar(i, energies_uj[i], yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=1.2, capsize=3, zorder=4)
        # Matches make_efficiency_figure()'s own label logic and decoder_comparison_4x2.py's
        # _decoder_display_label() exactly -- same three cases everywhere in this project.
        if rec["name"] == "speck":
            label = "SNN *speck"
        elif rec["name"] == "snn_pytorch":
            label = "SNN *PyTorch"
        else:
            label = rec["name"].upper()
        method = rec.get("energy_method") or "unknown"
        method_tag = {"rapl": "real (RAPL)", "chip_power_monitor": "real (chip)",
                      "proxy_psutil": "proxy"}.get(method, method)
        ax.annotate(f"{label}\n{method_tag}", (i, energies_uj[i]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([])  # names are already in the per-bar annotation above, incl. method
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.4)
    ax.margins(y=0.3)


def make_efficiency_figure_by_cohort(records, ci_bounds, laptop_cohort_names):
    """ONE figure -- two side-by-side panels when both measurement cohorts have data, a single
    panel when only one does -- reusing make_efficiency_figure()'s own scatter/legend/labeling
    logic for each panel (via its ax= parameter, see that function's own docstring) rather than
    a second, hand-duplicated copy of it.

    WHY TWO COHORTS, NOT ONE SHARED PLOT: KF/WF/LSTM/QRNN and test_all_decoders.py's own SNN
    row are profiled on Oscar; snn_pytorch/speck are infer_snn_speck.py's own laptop run. Both
    sides use the identical per-sample latency methodology now, but on DIFFERENT machines -- and
    the size of that difference is not hypothetical: the SNN, profiled on both, gave ~1.59
    ms/step on Oscar and ~0.48 ms/step on the laptop, a ~3.3x gap from machine speed alone
    (the laptop cannot install TensorFlow/sklearn to profile KF/WF/LSTM/QRNN itself and close
    this gap by moving everything onto one machine -- 1.9 GB free on a 62 GB disk). A single
    shared axis would invite exactly the comparisons that ~3.3x confound cannot support: LSTM/
    QRNN's own ~130x gap to the SNN is far too large to be a machine artifact and reads safely
    either way, but anything within a few x of that ~3.3x (e.g. WF vs the SNN) does not.

    Two PANELS in ONE saved image, not two separate files (unlike an earlier version of this
    function) -- this project's own energy figure already puts its own two (proxy vs. real)
    panels side by side in one image; matching that convention here keeps a reader flipping
    between fewer files while the cohort split itself still keeps the two groups visually and
    structurally distinct (separate titles, separate legends, each panel's own param-count
    reference sizes).

    Returns a single Figure. Raises RuntimeError if NEITHER cohort has any records (nothing to
    plot at all) -- unlike make_energy_bar_figure()'s own None-if-nothing-usable return, energy
    is allowed to be entirely absent from a decoder's record while RMSE/latency never are (see
    build_aggregate_records()'s own docstring), so an all-empty call here reflects a real
    upstream problem worth stopping on, not a normal "nothing to show" case.
    """
    oscar_records = [r for r in records if r["name"] not in laptop_cohort_names]
    laptop_records = [r for r in records if r["name"] in laptop_cohort_names]
    if not oscar_records and not laptop_records:
        raise RuntimeError("No decoder had both profiling and RMSE data -- nothing to plot.")

    cohorts = [c for c in [
        ("oscar", "Oscar (KF/WF/LSTM/QRNN and test_all_decoders.py's own SNN row)", oscar_records),
        ("laptop", "Speck-connected laptop (infer_snn_speck.py's own torch/speck run)", laptop_records),
    ] if c[2]]

    if len(cohorts) == 1:
        _, label, recs = cohorts[0]
        fig = make_efficiency_figure(recs, session_id=None)
        fig.axes[0].set_title(
            f"Decoding accuracy vs. computational cost -- {label}\n"
            f"(mean +/- 95% CI across sessions; marker size = mean param count)", fontsize=10)
        #add_error_bars(fig.axes[0], recs, ci_bounds)
        return fig

    print(f"\nNOTE: {len(cohorts)} measurement cohorts present -- see the two panels below. "
          f"Comparisons WITHIN one panel (same machine, same methodology) are directly "
          f"trustworthy; comparisons ACROSS panels are confounded by real, measured "
          f"machine-speed differences (~3.3x for this kind of workload) and should not be "
          f"read as purely algorithmic.")

    # sharex=True, sharey=True: both panels use ONE shared x-range (latency) and y-range
    # (RMSE), not each panel auto-fitting to only its own data. Same reasoning, and the same
    # matplotlib mechanism, as make_energy_bar_figure()'s own sharey -- a tightly-auto-scaled
    # panel makes small, likely-noise differences between its own two points (here: the laptop
    # panel's SNN *PyTorch vs SNN *speck) look dramatic, and hides where those points actually
    # sit relative to the Oscar panel's own much wider spread. In THIS project's actual data,
    # cohorts[0] (Oscar) has by far the wider natural range (KF at ~0.03ms to LSTM/QRNN at
    # ~64ms, RMSE ~27-45) and laptop's own two points fall comfortably inside it, so the
    # shared range matplotlib computes reduces to Oscar's own range unchanged -- effectively
    # "fixed by the left panel," exactly as asked. Implemented as a genuine shared range (the
    # union of whatever ends up plotted on EITHER panel), not a hard copy of the left panel's
    # own limits onto the right, specifically so a future session where the right panel's own
    # data falls outside today's left-panel range would EXPAND the shared view to keep showing
    # it, rather than silently clipping a real point off the edge of the plot.
    fig, axes = plt.subplots(1, len(cohorts), figsize=(7 * len(cohorts), 6),
                             sharex=True, sharey=True)
    axes = [axes] if len(cohorts) == 1 else list(axes)
    for ax, (_, label, recs) in zip(axes, cohorts):
        make_efficiency_figure(recs, session_id=None, ax=ax)
        ax.set_title(f"{label}\n(mean +/- 95% CI; marker size = mean param count)", fontsize=9)
        #add_error_bars(ax, recs, ci_bounds)
    fig.suptitle("Decoding accuracy vs. computational cost -- NOT comparable ACROSS panels, "
                 "see note above", fontsize=11, y=1.03)
    fig.tight_layout()
    return fig


def make_energy_bar_figure(records):
    """TWO independent panels -- 'Estimated (proxy, not measured)' and 'Measured (real)' --
    each with its OWN y-axis scale, NOT one shared axis with a dashed divider. Referenced by
    make_efficiency_figure()'s own docstring ("see plot_decoder_efficiency_aggregate.py's
    make_energy_bar_figure() for the figure that uses [energy_j/energy_method]") -- this is
    that function, previously missing despite the cross-reference already existing.

    WHY TWO PANELS, NOT ONE AXIS WITH A DIVIDER: a proxy estimate (elapsed_time x cpu_percent
    x an assumed TDP, from energy_meter.py's own fallback -- used here for every decoder
    Oscar's own RAPL-less nodes profile: KF/WF/LSTM/QRNN and Oscar's own SNN row) and a real
    RAPL/chip measurement are not the same quantity measured at different precision -- one is
    a rough software-side guess with no hardware instrumentation behind it at all, the other is
    what the hardware actually drew. Putting them on one y-axis, dashed line or not, still lets
    a reader's eye do an automatic height comparison the numbers don't support -- confirmed
    directly against this project's own real run: a real SNN *PyTorch bar (rapl) sat visually
    between two PROXY bars (KF, WF) on a shared log-scaled axis, which invites exactly the
    "chip-adjacent software is only modestly more expensive than a Kalman filter" reading the
    underlying numbers do not support. Two panels, two scales, removes the shared axis a
    reader's eye would otherwise use to compare them.

    A decoder appears in EXACTLY ONE panel, whichever its own recorded energy_method (see
    _REAL_ENERGY_METHODS) says it actually is -- never split, never shown twice.

    A BAR chart, not a second RMSE-vs-energy scatter: RMSE-vs-latency (make_efficiency_figure)
    and RMSE-vs-energy would tell nearly the same shape of story twice, since latency and
    energy are correlated for any one decoder (energy = power x latency). A bar chart of
    energy alone, decoder to decoder, is the figure that actually adds information beyond the
    existing scatter, and matches how this project has shown energy everywhere else
    (aggregate_speck_results.py's own energy_comparison.png).

    Decoders with NO usable energy (record['energy_j'] is None -- a --skip_energy profiling
    run, or every session's RAPL wraparound) are SKIPPED, not drawn as a zero-height bar --
    a missing measurement and a real zero both look like "nothing happened" on a bar chart,
    and only one of those is true. Returns None if NEITHER panel has anything to draw. If only
    ONE panel has data, returns a single-panel figure (an empty panel would be more confusing
    than informative, and there is nothing to protect a reader FROM if there is only one group).
    """
    usable = [r for r in records if r.get("energy_j") is not None]
    estimated = [r for r in usable if r.get("energy_method") not in _REAL_ENERGY_METHODS]
    measured = [r for r in usable if r.get("energy_method") in _REAL_ENERGY_METHODS]
    if not estimated and not measured:
        return None

    if estimated and measured:
        # sharey=True: both panels use the SAME y-axis ticks and limits (matplotlib auto-scales
        # to the UNION of what's plotted on either), rather than each auto-fitting to only its
        # own data. WHY THIS BELONGS HERE, not in tension with keeping the panels separate: two
        # independently-scaled panels can make bars in DIFFERENT panels look similarly sized
        # even when their real values differ by orders of magnitude, and -- the case that
        # actually motivated this -- can make a panel's OWN range look cleanly separated from
        # the other panel's when the underlying VALUES in fact overlap (confirmed directly: a
        # real run had 'Measured' bars sitting at ~1.4-3.1 uJ, a range that sat entirely INSIDE
        # 'Estimated' panel's own ~0.07-30 uJ span, invisible as long as each panel picked its
        # own axis). Sharing the scale surfaces that honestly while the panels, titles, and
        # hatching still keep proxy and real visually and structurally distinct -- the split
        # itself is still doing its job; this only fixes what "distinct" was silently implying
        # about relative MAGNITUDE, which was never the intent.
        fig, (ax_est, ax_meas) = plt.subplots(
            1, 2, figsize=(max(5.5, 1.3 * len(estimated)) + max(3.5, 1.3 * len(measured)), 5),
            sharey=True)
        _draw_energy_panel(ax_est, estimated, "Estimated (proxy, not measured)")
        _draw_energy_panel(ax_meas, measured, "Measured (real)")
        ax_est.set_ylabel("Mean energy per sample (uJ, log scale)")
        fig.suptitle("Energy per decoded sample -- proxy estimates and real measurements are "
                     "NOT directly comparable (see panel titles), but share ONE y-axis scale "
                     "here so relative magnitude is shown honestly",
                     fontsize=10, y=1.02)
    else:
        group, title = (estimated, "Estimated (proxy, not measured)") if estimated else (measured, "Measured (real)")
        fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(group)), 5))
        _draw_energy_panel(ax, group, "Energy per decoded sample\n" + title)
        ax.set_ylabel("Mean energy per sample (uJ, log scale)")

    fig.tight_layout()
    return fig


def main(args):
    decoders = [d.strip() for d in args.decoders.split(',')]

    per_decoder, energy_methods_seen = load_profiles(args.profiles_dir)
    rmses = load_rmse(args.combined_metrics_path, decoders)

    # "snn_pytorch" and "speck" appended ONLY if their own merge actually found real data (see
    # merge_impl_from_summary()'s own docstring) -- same graceful-degradation principle as
    # decoder_comparison_4x2.py: an experiment/subject without this data yet still produces
    # the original figure. "snn_pytorch" (infer_snn_speck.py's own "torch" run) merged FIRST,
    # matching decoder_comparison_4x2.py's own ordering, so it sits next to test_all_decoders.py's
    # differently-measured "snn" row rather than being an afterthought appended last.
    #
    # laptop_cohort_names tracks EXACTLY which decoders came from this merge -- i.e. were
    # measured on the speck-connected laptop -- as opposed to load_profiles()'s own decoders,
    # measured on Oscar. This is the split used below to build TWO separate RMSE-vs-latency
    # figures instead of one. WHY: a real, now-measured confound, not a hypothetical one --
    # the SAME per-sample methodology, applied to essentially the same SNN on both machines,
    # gave ~1.59 ms/step on Oscar and ~0.48 ms/step on the laptop -- a ~3.3x gap from machine
    # speed alone (confirmed directly: this laptop cannot install TensorFlow/sklearn to profile
    # KF/WF/LSTM/QRNN itself and put everything on one machine instead -- 1.9 GB free on a 62 GB
    # disk). Putting Oscar-measured and laptop-measured LATENCY on one shared x-axis would invite
    # exactly the same false comparison the energy panels (see make_energy_bar_figure()) already
    # guard against -- just for a different reason (machine speed here, real-vs-proxy there).
    # Decoders within EITHER group remain fully comparable to each other (same machine, same
    # methodology); only comparisons that cross this split should be read with real caution.
    laptop_cohort_names = set()
    if merge_impl_from_summary("torch", "snn_pytorch", args.speck_summary_path, per_decoder,
                               rmses, energy_methods_seen):
        decoders.append("snn_pytorch")
        laptop_cohort_names.add("snn_pytorch")
    if merge_impl_from_summary("speck", "speck", args.speck_summary_path, per_decoder,
                               rmses, energy_methods_seen):
        decoders.append("speck")
        laptop_cohort_names.add("speck")

    check_param_count_consistency(per_decoder)

    records, ci_bounds = build_aggregate_records(per_decoder, rmses, decoders, energy_methods_seen)
    if not records:
        raise RuntimeError("No decoder had both profiling and RMSE data -- nothing to plot.")

    print("Aggregate records (mean across sessions):")
    for r in records:
        n_lat = len(per_decoder[r["name"]]["latency_s"])
        n_rmse = len(rmses[r["name"]])
        if r["energy_j"] is not None:
            method_tag = {"rapl": "real RAPL", "chip_power_monitor": "real chip",
                          "proxy_psutil": "PROXY ESTIMATE"}.get(r["energy_method"], r["energy_method"])
            energy_str = f"energy={r['energy_j']*1e6:.3f} uJ/sample (n={r['n_energy_sessions']}, {method_tag})"
        else:
            energy_str = "energy=n/a (no usable energy_j in any session's profile)"
        print(f"  {r['name']:>5s} | RMSE={r['rmse']:.2f} (n={n_rmse} sessions) | "
              f"latency={r['latency_s']*1000:.4f} ms/sample (n={n_lat} sessions) | "
              f"params={r['param_count']:,.0f} | {energy_str}")

    # TWO separate RMSE-vs-latency figures, one per measurement COHORT (see laptop_cohort_names'
    # own comment above for why), each built from make_efficiency_figure()/add_error_bars()
    # completely UNCHANGED -- reused exactly as before, just called once per cohort's own
    # record subset instead of once over everything, so nothing about how a single figure is
    # drawn needed to change, only which records go into which call. A cohort with zero
    # records is skipped entirely (no empty figure saved) -- e.g. before any speck_summary is
    # merged, only the Oscar cohort exists, and that alone should still produce output.
    cohort_fig = make_efficiency_figure_by_cohort(records, ci_bounds, laptop_cohort_names)

    energy_fig = make_energy_bar_figure(records)  # already its own proxy-vs-real split, see
    # that function's own docstring -- independent of, and not merged with, the cohort split
    # above (the two happen to align on every dataset seen so far, but are conceptually
    # separate: one is about which MACHINE measured a number, the other about whether that
    # number is a real measurement or a proxy estimate).

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        cohort_fig.savefig(args.save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.save_path}")
        if energy_fig is not None:
            base, ext = os.path.splitext(args.save_path)
            energy_save_path = f"{base}_energy{ext}"
            energy_fig.savefig(energy_save_path, dpi=150, bbox_inches='tight')
            print(f"Saved energy figure to {energy_save_path}")
        else:
            print("No decoder had usable energy data -- energy figure skipped.")
    else:
        plt.show()
        if energy_fig is not None:
            plt.show()

    if args.summary_save_path:
        os.makedirs(os.path.dirname(args.summary_save_path) or '.', exist_ok=True)
        summary = {"decoders": {
            r["name"]: {"rmse_mean": r["rmse"], "latency_s_mean": r["latency_s"],
                       "param_count_mean": r["param_count"], "n_sessions": r["n_sessions"],
                       "energy_j_mean": r["energy_j"], "energy_j_lo": r["energy_lo"],
                       "energy_j_hi": r["energy_hi"], "energy_method": r["energy_method"],
                       "n_energy_sessions": r["n_energy_sessions"],
                       # "oscar" or "laptop" -- see laptop_cohort_names' own comment above for
                       # why latency/energy are only directly comparable WITHIN one cohort.
                       "latency_cohort": "laptop" if r["name"] in laptop_cohort_names else "oscar"}
            for r in records}}
        with open(args.summary_save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to {args.summary_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles_dir', type=str, required=True,
                         help='Directory of *_profile.json files, one per session, written by '
                              'plot_decoder_efficiency.py --profile_save_path')
    parser.add_argument('--combined_metrics_path', type=str, required=True)
    parser.add_argument('--decoders', type=str, default='lstm,qrnn,kf,wf,snn')
    parser.add_argument('--speck_summary_path', type=str, default='speck_summary/aggregated_summary.json',
                         help="aggregate_speck_results.py's own --output-dir/aggregated_summary.json "
                              "output. If missing, 'SNN *speck' is skipped and the original decoder "
                              "comparison is still produced (see merge_speck_summary()).")
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--summary_save_path', type=str, default=None,
                         help="Optional: also dump the final per-decoder aggregate (RMSE, "
                              "latency, param_count, energy, energy_method, n_sessions) as "
                              "JSON here -- for anything downstream that wants the numbers "
                              "without re-parsing this script's console output.")
    args = parser.parse_args()
    main(args)
