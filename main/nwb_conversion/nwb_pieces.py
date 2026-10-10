"""
Sessions recorded in several runs (the multi-run Nitschke NWB files).

Some Nitschke sessions concatenate separate recording runs into one file, each
with its own clock, so the Hand timestamps (and every unit's spike times)
step backward at each run boundary, and the runs' clock ranges overlap
(several runs cover 6,000-12,000 s). Selecting a trial's samples by clock
time alone mixes runs, and interpolating that mix creates the velocity
spikes; the trials table's stored per-trial row slices (trials.timeseries)
point to the wrong run for most later trials, so they are not used either.

Terms: a PIECE is a stretch of an array whose timestamps never step back; a
RUN is a stretch of consecutive trial rows whose start_time does not drop by
more than RUN_RESET_S.

Trial -> hand piece (assign_trials):
  - the candidates are the pieces whose clock covers the trial's
    [start_time, stop_time];
  - one candidate: that piece;
  - several: at the end of a reach the hand sits on the target, so the hand
    position at move_ends_time (else stop_time) is compared with
    hit_target_position in each candidate, after a linear hand -> target fit
    per axis on the trials with one candidate. The run's majority piece
    (among its trials with a clear winner) is taken if it is a candidate and
    lands within MAX_DIST_FACTOR x the median calibration distance, else the
    trial's own clear winner, else the trial is skipped.

Spikes (SpikePieces): hand pieces whose clocks continue each other (the
next starts no more than JOIN_S before the previous ends: a tiny backward
step inside one run) form one hand GROUP. Every unit's spike times are cut
into pieces the same way and each spike piece is given, in recording order,
to the group that contains most of it; a trial's spikes are its group's.
Where two spike pieces of one group overlap in time, a trial window
touching the overlap is ambiguous and skipped. In some sessions units 0 and 1 are cut at the wrong place (unit 0
holds the start of unit 0 only, unit 1 the rest of unit 0 and all of unit
1); repair_units_0_1 re-cuts them.

A session with one hand piece needs none of this: every trial is assigned
to it, and the converter's output is the same as before.
"""

from collections import Counter

import numpy as np

RUN_RESET_S = 600.0       # a run starts where start_time drops by more than this
COVER_TOL_S = 0.5         # a piece covers a trial if its clock range holds [start, stop] within this
NEAR_S = 0.01             # hand sample used for a time must lie within this of it
MIN_CALIBRATION = 20      # single-candidate trials needed for the hand -> target fit
CLEAR_FACTOR = 3.0        # clear winner: best < 3 x calibration median and runner-up > 3 x best
MAX_DIST_FACTOR = 5.0     # a chosen piece must land within 5 x the calibration median
JOIN_S = 1.0              # hand pieces whose clocks continue within this share their spikes


def split_pieces(t):
    """[(first_row, stop_row)] of the stretches of t that never step back."""
    t = np.asarray(t)
    edges = np.r_[0, np.flatnonzero(np.diff(t) < 0) + 1, len(t)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def trial_runs(start_times, reset_s=RUN_RESET_S):
    """Run index of every trial row (a new run where start_time drops by more
    than reset_s below the running maximum)."""
    runs = np.zeros(len(start_times), dtype=int)
    run, top = 0, -np.inf
    for i, s in enumerate(start_times):
        if s < top - reset_s:
            run, top = run + 1, s
        top = max(top, s)
        runs[i] = run
    return runs


def _value_at(t, values, when, tol=NEAR_S):
    """values at the sample of t nearest each time in when (NaN if none within tol)."""
    out = np.full((len(when),) + values.shape[1:], np.nan)
    j = np.clip(np.searchsorted(t, when), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - when) <= np.abs(t[j] - when), j - 1, j)
    ok = np.abs(t[j] - when) <= tol
    out[ok] = values[j[ok]]
    return out


def assign_trials(hand_t, hand_xy, pieces, trials):
    """(piece index per trial, -1 if skipped; per-trial reason; summary dict).

    trials: DataFrame with start_time, stop_time and, for sessions of several
    pieces, hit_target_position (move_ends_time used when present)."""
    start = trials["start_time"].to_numpy(float)
    stop = trials["stop_time"].to_numpy(float)
    n = len(start)
    if len(pieces) == 1:
        return np.zeros(n, dtype=int), np.array(["single piece"] * n, dtype=object), {"n_pieces": 1}

    ranges = [(hand_t[a], hand_t[b - 1]) for a, b in pieces]
    cover = np.array([[(lo - COVER_TOL_S <= s) and (e <= hi + COVER_TOL_S) for lo, hi in ranges]
                      for s, e in zip(start, stop)], dtype=bool)
    n_cover = cover.sum(axis=1)
    piece = np.full(n, -1)
    reason = np.array(["no piece covers the trial"] * n, dtype=object)
    one = np.flatnonzero(n_cover == 1)
    piece[one] = cover[one].argmax(axis=1)
    reason[one] = "one candidate"
    runs = trial_runs(start)
    summary = {"n_pieces": len(pieces), "piece_ranges_s": [[float(lo), float(hi)] for lo, hi in ranges],
               "n_runs": int(runs.max() + 1), "n_trials_by_candidates": dict(Counter(map(int, n_cover)))}

    ambiguous = np.flatnonzero(n_cover >= 2)
    if len(ambiguous) == 0:
        return piece, reason, summary
    if "hit_target_position" not in trials:
        reason[ambiguous] = "several candidate pieces, no hit_target_position"
        return piece, reason, summary

    when = stop.copy()
    if "move_ends_time" in trials:
        me = trials["move_ends_time"].to_numpy(float)
        when = np.where(np.isfinite(me) & (me > 0), me, stop)
    target = np.vstack([np.asarray(v, float)[:2] for v in trials["hit_target_position"]])
    hand_end = np.stack([_value_at(hand_t[a:b], hand_xy[a:b], when) for a, b in pieces], axis=1)
    hand_end[~cover] = np.nan                                              # (n, pieces, 2)

    calib = one[np.isfinite(hand_end[one, piece[one], 0]) & np.isfinite(target[one]).all(axis=1)]
    if len(calib) < MIN_CALIBRATION:
        reason[ambiguous] = f"several candidate pieces, only {len(calib)} trials to calibrate"
        return piece, reason, summary
    h = hand_end[calib, piece[calib]]
    fit = np.array([np.polyfit(h[:, k], target[calib, k], 1) for k in range(2)])   # [[slope, offset]] per axis
    dist = np.linalg.norm(hand_end * fit[:, 0] + fit[:, 1] - target[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    cal = float(np.median(dist[calib, piece[calib]]))
    limit = MAX_DIST_FACTOR * cal
    order = np.sort(dist, axis=1)
    winner = np.argmin(dist, axis=1)
    clear = (order[:, 0] < CLEAR_FACTOR * cal) & (order[:, 1] > CLEAR_FACTOR * order[:, 0])

    majority = {}
    for r in np.unique(runs[ambiguous]):
        votes = Counter(winner[i] for i in ambiguous if runs[i] == r and clear[i])
        if votes:
            majority[int(r)] = int(votes.most_common(1)[0][0])
    n_disagree = 0
    for i in ambiguous:
        m = majority.get(int(runs[i]))
        if m is not None and cover[i, m] and dist[i, m] <= limit:
            piece[i], reason[i] = m, "run majority"
            n_disagree += int(clear[i] and winner[i] != m)
        elif clear[i] and dist[i, winner[i]] <= limit:
            piece[i], reason[i] = winner[i], "clear winner"
        else:
            reason[i] = "several candidate pieces, no clear match to the target"
    resolved = ambiguous[piece[ambiguous] >= 0]
    resolved_dist = dist[resolved, piece[resolved]]
    summary.update({
        "calibration": {"n_trials": int(len(calib)), "slope_offset_xy": fit.tolist(),
                        "median_distance": cal, "max_distance_kept": limit},
        "run_majority_piece": majority,
        "n_ambiguous": int(len(ambiguous)), "n_resolved": int(len(resolved)),
        "n_run_majority_overriding_trial_winner": n_disagree,
        "resolved_distance_median": float(np.median(resolved_dist)) if len(resolved) else None,
    })
    return piece, reason, summary


def repair_units_0_1(unit_spike_times):
    """Re-cut units 0 and 1 where they are cut at the wrong place: unit 0 has
    fewer pieces than the typical unit and unit 1 more, and their pooled
    spikes cut into exactly twice the typical count. Returns (units, note).
    Raises ValueError for that signature without the twice-typical count."""
    counts = [len(split_pieces(u)) for u in unit_spike_times]
    if len(counts) < 2:
        return unit_spike_times, None
    typical = Counter(counts).most_common(1)[0][0]
    if not (counts[0] < typical < counts[1]):
        return unit_spike_times, None
    pooled = np.concatenate([unit_spike_times[0], unit_spike_times[1]])
    cuts = split_pieces(pooled)
    if len(cuts) != 2 * typical:
        raise ValueError(f"units 0 and 1 look mis-cut ({counts[0]} and {counts[1]} pieces, typical "
                         f"{typical}), but pooled they make {len(cuts)} pieces, not {2 * typical}")
    split = cuts[typical][0]
    units = list(unit_spike_times)
    units[0], units[1] = pooled[:split], pooled[split:]
    return units, (f"units 0 and 1 re-cut: {counts[0]} + {counts[1]} pieces -> {typical} + {typical}")


def hand_groups(hand_ranges, join_s=JOIN_S):
    """Group index of every hand piece: consecutive pieces whose clocks continue
    (next start >= previous end - join_s) share a group."""
    groups = [0]
    for (_, prev_hi), (lo, _) in zip(hand_ranges[:-1], hand_ranges[1:]):
        groups.append(groups[-1] + int(lo < prev_hi - join_s))
    return groups


class SpikePieces:
    """Every unit's spike pieces, each given to a hand group in recording order."""

    def __init__(self, unit_spike_times, hand_ranges):
        self.group_of = hand_groups(hand_ranges)
        n_groups = self.group_of[-1] + 1
        group_ranges = [(min(lo for (lo, _), g in zip(hand_ranges, self.group_of) if g == k),
                         max(hi for (_, hi), g in zip(hand_ranges, self.group_of) if g == k))
                        for k in range(n_groups)]
        self.units = []                          # per unit: [(group, spikes)]
        self.min_containment = 1.0
        self.ambiguous = [[] for _ in range(n_groups)]   # per group: overlapping [lo, hi]
        for spk in unit_spike_times:
            spk = np.asarray(spk, dtype=np.float64)
            assigned, j = [], 0
            for a, b in split_pieces(spk):
                piece = spk[a:b]
                frac = [np.mean((piece >= lo) & (piece <= hi)) for lo, hi in group_ranges[j:]]
                k = j + int(np.argmax(frac))            # first of the best: ties go to the earlier group
                self.min_containment = min(self.min_containment, float(frac[k - j]))
                assigned.append((k, piece))
                j = k
            for (p, x), (q, y) in zip(assigned[:-1], assigned[1:]):
                if p == q and y[0] <= x[-1]:            # same group, clocks overlap
                    self.ambiguous[p].append((float(y[0]), float(x[-1])))
            self.units.append(assigned)

    def window_is_ambiguous(self, hand_piece, t0, t1):
        return any(lo <= t1 and t0 <= hi for lo, hi in self.ambiguous[self.group_of[hand_piece]])

    def trial_spikes(self, hand_piece):
        """Every unit's spike times in this hand piece's group (pieces
        concatenated in order), to be cut to a trial's window by the caller."""
        g = self.group_of[hand_piece]
        out = []
        for assigned in self.units:
            parts = [x for p, x in assigned if p == g]
            out.append(parts[0] if len(parts) == 1 else
                       np.sort(np.concatenate(parts)) if parts else np.zeros(0))
        return out

    def summary(self):
        per_group = Counter(p for assigned in self.units for p, _ in assigned)
        return {"hand_group_of_piece": list(self.group_of),
                "spike_pieces_per_group": {int(k): int(v) for k, v in sorted(per_group.items())},
                "min_containment": self.min_containment,
                "overlap_intervals_s": [[list(iv) for iv in sorted(set(a))] for a in self.ambiguous]}
