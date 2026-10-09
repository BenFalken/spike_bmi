"""
Removes tracking glitches from raw hand-position traces BEFORE velocity is derived from them.

WHY THIS EXISTS: convert_nwb_trials_to_raw_h5.py computes velocity as diff(position)/dt. The
Nitschke recordings contain position jumps (a few hundred per session) that turn into
single-sample velocity spikes of ~60,000-77,000 units/s, versus a real 99.5th-percentile speed of
~865 (diagnose_hkm_sessions.py). Those spikes dominate every trial-level RMSE, the training loss
and the least-squares fits of KF/WF, so no decoder can learn on the affected sessions.

METHOD (on the NATIVE position samples, before resampling to 250 Hz):
  1. speed between consecutive native samples = |dxy| / dt
  2. any interval faster than `max_speed` (or with non-finite / non-positive dt) is a glitch
  3. the displacement across each glitch interval is replaced by what the hand was doing around it
     (velocity linearly interpolated from the nearest good intervals, times dt) and position is
     rebuilt by cumulative sum -> the trace stays continuous, with no offset left behind.
     This handles both glitch shapes without having to know which one occurs:
       * out-and-back (position jumps away for a sample and returns): both jumps are replaced
       * step (position jumps and stays): the step is replaced, later samples are shifted back
GUARANTEE: after this, no native interval exceeds `max_speed`, and the 250 Hz grid velocity is an
average of native velocities, so |v| on the grid is also <= `max_speed`.
Real motion is untouched unless it is faster than `max_speed` (default 3000 units/s, ~3.5x the
real 99.5th-percentile speed). A false positive only removes a tiny displacement (one native
interval, ~1.5 ms), so over-flagging is harmless.
"""
import numpy as np

DEFAULT_MAX_SPEED = 3000.0


def despike_position(t, xy, max_speed=DEFAULT_MAX_SPEED):
    """t: (n,) timestamps (s); xy: (n, 2) positions.
    Returns (xy_clean, info). max_speed <= 0 or None disables cleaning."""
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    info = {"n_glitch_intervals": 0, "glitch_speeds": [], "glitch_displacements": [],
            "glitch_native_indices": []}
    if not max_speed or max_speed <= 0 or len(t) < 2:
        return xy.copy(), info

    dt = np.diff(t)
    d = np.diff(xy, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(d, axis=1) / dt
    bad = (~np.isfinite(speed)) | (dt <= 0) | (speed > max_speed)
    if not bad.any():
        return xy.copy(), info

    # Replace each glitch interval's displacement with interpolated-velocity * dt.
    d_clean = d.copy()
    good = ~bad
    if good.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            v_good = d[good] / dt[good][:, None]
        idx_all = np.arange(len(dt))
        v_fill = np.stack([np.interp(idx_all[bad], idx_all[good], v_good[:, k]) for k in range(2)], axis=1)
        dt_bad = np.where(dt[bad] > 0, dt[bad], 0.0)
        d_clean[bad] = v_fill * dt_bad[:, None]
    else:
        d_clean[bad] = 0.0
    xy_clean = np.empty_like(xy)
    xy_clean[0] = np.nan_to_num(xy[0])
    xy_clean[1:] = xy_clean[0] + np.cumsum(d_clean, axis=0)

    idx = np.where(bad)[0]
    info["n_glitch_intervals"] = int(len(idx))
    info["glitch_native_indices"] = [int(i) for i in idx]
    info["glitch_speeds"] = [float(speed[i]) if np.isfinite(speed[i]) else float("inf") for i in idx]
    info["glitch_displacements"] = [float(np.linalg.norm(d[i])) if np.isfinite(d[i]).all() else float("nan")
                                    for i in idx]
    return xy_clean, info
