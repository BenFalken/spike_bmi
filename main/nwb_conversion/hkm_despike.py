"""
Remove hand-tracking glitches from a trial's native position samples, before
velocity is derived from them (convert_nwb_trials_to_raw_h5.py).

Velocity is diff(position) / dt, so a position jump of a few hundred units
between two native samples becomes a velocity spike of ~60,000-77,000
units/s; real movement stays below ~865 units/s (99.5th percentile). The
Nitschke sessions hold a few hundred such jumps each, enough to dominate
every RMSE and every least-squares fit.

Method: an interval between consecutive native samples is a glitch if its
speed exceeds max_speed (or its dt is not positive). Its displacement is
replaced by the velocity interpolated from the neighbouring good intervals
times its dt, and position is rebuilt by cumulative sum, so the trace stays
continuous and no offset is left behind. This handles both glitch shapes:
out-and-back jumps (both jumps replaced) and steps (the step replaced, later
samples shifted back).

After cleaning no native interval is faster than max_speed, and the 250 Hz
grid velocity, a weighted average of native velocities, is not either. The
default 3000 units/s is ~3.5x the real 99.5th percentile; a false positive
only replaces one ~1.5 ms native interval.
"""

import numpy as np

DEFAULT_MAX_SPEED = 3000.0


def despike_position(t, xy, max_speed=DEFAULT_MAX_SPEED):
    """t (n,) timestamps in s, xy (n, 2) positions. Returns (xy_clean, info);
    info lists the glitch intervals. max_speed of 0 or None disables cleaning."""
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    info = {"n_glitch_intervals": 0, "glitch_speeds": [], "glitch_native_indices": []}
    if not max_speed or max_speed <= 0 or len(t) < 2:
        return xy.copy(), info

    dt = np.diff(t)
    d = np.diff(xy, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(d, axis=1) / dt
    bad = ~np.isfinite(speed) | (dt <= 0) | (speed > max_speed)
    if not bad.any():
        return xy.copy(), info

    d_clean = d.copy()
    good = ~bad
    if good.any():
        idx = np.arange(len(dt))
        v_good = d[good] / dt[good][:, None]
        v_fill = np.stack([np.interp(idx[bad], idx[good], v_good[:, k]) for k in range(2)], axis=1)
        d_clean[bad] = v_fill * np.where(dt[bad] > 0, dt[bad], 0.0)[:, None]
    else:
        d_clean[bad] = 0.0
    xy_clean = np.empty_like(xy)
    xy_clean[0] = np.nan_to_num(xy[0])
    xy_clean[1:] = xy_clean[0] + np.cumsum(d_clean, axis=0)

    bad_idx = np.flatnonzero(bad)
    info["n_glitch_intervals"] = int(len(bad_idx))
    info["glitch_native_indices"] = [int(i) for i in bad_idx]
    info["glitch_speeds"] = [float(speed[i]) if np.isfinite(speed[i]) else float("inf") for i in bad_idx]
    return xy_clean, info
