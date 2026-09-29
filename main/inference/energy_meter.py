"""
EnergyMeter: a small, dependency-light class for measuring wall-clock
latency and CPU energy around a block of code.

Extracted into its own module specifically so scripts that only need
energy measurement -- e.g. infer_snn_speck.py's host-side (torch/
discretized/specksim) timing -- don't have to import
test_all_decoders.py's full machinery (torch, h5py, sklearn,
bmi.decoders, sinabs, plus that module's own CUDA_VISIBLE_DEVICES-
forcing) just to get this one class. test_all_decoders.py and
plot_decoder_efficiency.py now both import EnergyMeter FROM HERE too,
rather than test_all_decoders.py defining it and everyone else reaching
through that heavier module to get it.

This module's only real dependency is psutil -- no numpy, no torch,
nothing that takes meaningful time or memory to import, and no import-
time side effects beyond the RAPL availability check and thread-pinning
below (both deliberate, see their own docstrings).

--------------------------------------------------------------------
IMPORT-ORDER WARNING -- read this before adding this import to a new
script: pin_math_library_threads() (called automatically at THIS
module's own import time, below) only works if it runs BEFORE numpy/
scipy/torch/etc. are first imported anywhere in the process -- OpenBLAS/
MKL/OMP read their thread-count env vars once, at their own
import/first-use time, not on every call. Importing energy_meter AFTER
`import torch` or `import numpy` elsewhere in your script's import list
is TOO LATE: those libraries already initialized their thread pools
under whatever the env vars said at THAT point (unset, in the common
case, meaning "use every core the OS reports"), and setting the env var
afterward does nothing retroactively.

Put `from energy_meter import EnergyMeter` at the VERY TOP of your
script's imports, before torch/numpy/scipy/sklearn/anything BLAS-backed.
CONFIRMED this was actually being violated: the previous revision of
infer_snn_speck.py imported EnergyMeter (then still living inside
test_all_decoders.py) AFTER its own `import torch` a few lines above --
the thread-pinning was silently a no-op there the whole time. This
module's extraction fixes that as a side effect, but only if the import
itself moves to the top of the importing script too -- moving where
EnergyMeter is DEFINED doesn't help if the import statement stays in
the same (too-late) place it was before.
--------------------------------------------------------------------
"""

import glob
import os
import time

import psutil

# --------------------------------------------------------------------------
# Thread pinning
# --------------------------------------------------------------------------
# Pin every math library's thread count to this job's actual core
# allocation, at IMPORT TIME of this module (see the import-order
# warning above for why that has to be early). OpenBLAS, MKL, and OMP
# often default to detecting the WHOLE NODE's core count rather than the
# Slurm cgroup a given task is actually confined to -- left unpinned, a
# single call into numpy/scipy/torch's BLAS backend can spin up far more
# threads than --cpus-per-task, which has two real consequences:
#   - LATENCY gets noisier: those extra threads mostly contend for the
#     same handful of real cores rather than adding genuine parallelism.
#   - EnergyMeter's proxy_psutil fallback becomes badly inflated:
#     psutil's cpu_percent() is per-process and uncapped at 100% -- it's
#     normalized to ONE logical core, so N contending threads report
#     ~N*100%, and energy_j = ASSUMED_CPU_TDP_WATTS * cpu_frac * latency_s
#     inflates by that same factor even though the node isn't actually
#     drawing that much power. Confirmed in practice (see
#     diagnose_energy_proxy.py) -- KF and SNN workloads showed
#     implied_cores in the 20-30 range on a --cpus-per-task=2 allocation
#     before this fix, vs. ~1-2 after.
#
# ENERGY_METER_NUM_THREADS is a manual override (e.g. running outside
# Slurm); TEST_ALL_DECODERS_NUM_THREADS is also still honored, for
# backward compatibility with scripts written before this module
# existed. SLURM_CPUS_PER_TASK (set automatically by Slurm) is the
# fallback if neither override is set. If NONE of these are set, this
# deliberately does nothing and prints a warning -- silently picking a
# number here would just trade one unexamined assumption for another;
# better to make the missing config visible.


def pin_math_library_threads():
    """Sets OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/
    NUMEXPR_NUM_THREADS/VECLIB_MAXIMUM_THREADS from (in priority order)
    ENERGY_METER_NUM_THREADS, TEST_ALL_DECODERS_NUM_THREADS (back-compat),
    or SLURM_CPUS_PER_TASK. Called automatically once at this module's
    own import time -- see the import-order warning at the top of this
    file for why that has to happen before numpy/torch/etc. are
    imported. Safe to call again manually later (idempotent -- just
    re-reads and re-sets), though that won't retroactively re-pin
    libraries that already initialized their thread pools. Returns the
    thread count used, or None if nothing was set.
    """
    override = os.environ.get("ENERGY_METER_NUM_THREADS") or os.environ.get(
        "TEST_ALL_DECODERS_NUM_THREADS")
    source = ("ENERGY_METER_NUM_THREADS" if os.environ.get("ENERGY_METER_NUM_THREADS")
              else "TEST_ALL_DECODERS_NUM_THREADS" if override else "SLURM_CPUS_PER_TASK")
    n_threads_env = override or os.environ.get("SLURM_CPUS_PER_TASK")
    if not n_threads_env:
        print("WARNING: none of ENERGY_METER_NUM_THREADS, TEST_ALL_DECODERS_NUM_THREADS, "
              "or SLURM_CPUS_PER_TASK is set -- OpenBLAS/MKL/OMP will use their own default "
              "thread count (often the WHOLE node's core count), which can make both latency "
              "and proxy_psutil energy_j numbers noisy or inflated if this process doesn't "
              "actually have the whole node to itself. Set ENERGY_METER_NUM_THREADS "
              "explicitly if running outside Slurm.")
        return None
    n_threads = int(n_threads_env)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"Pinning math library thread counts to {n_threads} (from {source})")
    return n_threads


NUM_THREADS = pin_math_library_threads()  # runs at import time -- see warning above


def pin_torch_threads():
    """Explicitly pins PyTorch's own intra-op thread pool too -- belt
    and suspenders alongside the env vars above, since torch's thread
    pool doesn't reliably derive from OMP_NUM_THREADS on every build/
    platform. Call this AFTER `import torch` in your own script -- torch
    is deliberately NOT imported by this module itself (see this
    module's own docstring: scripts that don't need torch shouldn't pay
    for importing it just to get EnergyMeter). No-op if NUM_THREADS is
    None (nothing to pin to -- see pin_math_library_threads()'s own
    warning, which already fired at import time in that case).
    """
    if NUM_THREADS is None:
        return
    import torch
    torch.set_num_threads(NUM_THREADS)


# --------------------------------------------------------------------------
# Energy measurement
# --------------------------------------------------------------------------
# "Energy" here means CPU package energy, measured via Intel RAPL
# (/sys/class/powercap/intel-rapl) when the node exposes it -- checked
# ONCE at this module's import time, not per call, since the sysfs paths
# don't change mid-run. Falls back to a psutil-based PROXY (elapsed_time
# x this process's CPU utilization x an assumed constant TDP) when RAPL
# isn't readable, and every energy_j value is tagged with which of the
# two produced it (see EnergyMeter.energy_method) so a proxy number is
# never later confused with a real measurement.
#
# CAVEAT even when RAPL IS available: the RAPL counter measures the
# WHOLE physical CPU package's energy draw, not just this process's
# cores. On an exclusively-allocated node this is fine; on a node shared
# with other processes' cores on the same package, readings will
# include their draw too.

def _discover_rapl_package_paths():
    """Every intel-rapl:N/energy_uj path whose SIBLING intel-rapl:N/name file reads
    'package-<int>' -- i.e. an actual CPU package domain, and nothing else.

    BUG FIX: the previous version of this function was a bare glob,
    'intel-rapl:*/energy_uj', with no check on what each domain actually IS. On modern
    Intel client/laptop platforms that ALSO exposes a 'psys' domain alongside 'package-N'
    -- Intel's own PSys/"Platform Power" rail, which covers the WHOLE laptop (display,
    integrated GPU, WiFi, chipset -- everything, not just the CPU) -- the bare glob
    matched and SUMMED that domain in too, despite this module's own docstring stating
    the intent as 'CPU package energy'. Confirmed directly on real hardware: a laptop
    with intel-rapl:0/name='package-0' and intel-rapl:1/name='psys' reported host-CPU
    'torch' power in the 71-78W range across several real runs of a single-threaded,
    ~1M-op/s workload -- physically implausible as that workload's own draw, and
    unexplained by system-wide CPU utilization (confirmed separately at 5.6% during one
    of those runs) -- consistent with a roughly-constant, workload-independent psys
    baseline (display/WiFi/chipset draw doesn't track what one pinned thread is doing)
    dominating a sum that was never supposed to include it.

    A domain whose /name file is missing or unreadable is EXCLUDED, not included --
    on ambiguity, err toward under-counting (a package domain this function fails to
    find just means _RAPL_AVAILABLE is False and the proxy fallback kicks in, which is
    visible and labeled) rather than silently summing an unidentified rail into
    something labeled 'CPU package energy'.

    Server/HPC platforms (e.g. Oscar) are not generally known to expose a psys domain
    at all (it is a client/laptop platform feature) -- this filter should be a no-op
    there (every domain found is already named 'package-N'), but is applied
    unconditionally rather than only on machines suspected of having the problem,
    since the whole point is not needing to suspect it in the first place.
    """
    candidates = sorted(glob.glob('/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj'))
    package_paths, excluded = [], []
    for p in candidates:
        name_path = os.path.join(os.path.dirname(p), 'name')
        try:
            with open(name_path) as f:
                name = f.read().strip()
        except OSError:
            excluded.append((p, '<name unreadable>'))
            continue
        if name.startswith('package-') and name[len('package-'):].isdigit():
            package_paths.append(p)
        else:
            excluded.append((p, name))
    return package_paths, excluded


_RAPL_ENERGY_PATHS, _RAPL_EXCLUDED_DOMAINS = _discover_rapl_package_paths()

ASSUMED_CPU_TDP_WATTS = 65.0  # only used by the proxy fallback -- a rough
                              # generic-server-CPU package TDP, NOT a
                              # measurement of any particular machine


def _read_rapl_energy_uj():
    """Sum of all CPU PACKAGE RAPL energy counters (see _discover_rapl_package_paths()
    for exactly which domains that is and, critically, which it deliberately excludes),
    in microjoules, or None if unreadable (missing files, or a permission/read error) --
    callers must check for None and fall back rather than silently guessing a value."""
    if not _RAPL_ENERGY_PATHS:
        return None
    total = 0
    try:
        for p in _RAPL_ENERGY_PATHS:
            with open(p) as f:
                total += int(f.read().strip())
    except (OSError, ValueError):
        return None
    return total


_RAPL_AVAILABLE = _read_rapl_energy_uj() is not None
if _RAPL_AVAILABLE:
    print(f"Energy measurement: using Intel RAPL ({len(_RAPL_ENERGY_PATHS)} package "
          f"domain(s) found under /sys/class/powercap/intel-rapl) -- real CPU package "
          f"energy in joules.")
    if _RAPL_EXCLUDED_DOMAINS:
        print(f"  Excluded {len(_RAPL_EXCLUDED_DOMAINS)} non-package RAPL domain(s), not "
              f"summed into the reading above: "
              + ", ".join(f"{name}" for _, name in _RAPL_EXCLUDED_DOMAINS)
              + " (e.g. 'psys' is Intel's whole-PLATFORM rail -- display, integrated GPU, "
                "WiFi, chipset -- not CPU-specific; see _discover_rapl_package_paths()'s "
                "own docstring).")
else:
    print("Energy measurement: RAPL energy counters not readable on this node (no "
          "/sys/class/powercap/intel-rapl/*/energy_uj with a 'package-N' name, or no "
          "permission to read them) -- "
          "falling back to a PROXY (elapsed_time x process CPU utilization x an assumed "
          f"{ASSUMED_CPU_TDP_WATTS:.0f}W TDP). Treat these as rough RELATIVE estimates for "
          "comparing workloads against each other, not as real joules.")


class EnergyMeter:
    """Context manager measuring wall time (s) and CPU energy (J) consumed
    by the wrapped block.

    Sets .latency_s, .energy_j, .energy_method ('rapl' or 'proxy_psutil')
    on exit. energy_j is None (not a guessed value) if a RAPL counter
    wraparound is detected (it's a fixed-width counter that periodically
    resets) mid-measurement.
    """

    def __enter__(self):
        self._t0 = time.perf_counter()
        if _RAPL_AVAILABLE:
            self._e0_uj = _read_rapl_energy_uj()
        else:
            self._proc = psutil.Process()
            self._proc.cpu_percent(interval=None)  # prime; first call is a no-op baseline
        return self

    def __exit__(self, *exc_info):
        self.latency_s = time.perf_counter() - self._t0
        if _RAPL_AVAILABLE:
            e1_uj = _read_rapl_energy_uj()
            delta_uj = None if e1_uj is None else e1_uj - self._e0_uj
            if delta_uj is not None and delta_uj < 0:
                print("  WARNING: RAPL energy counter wrapped around mid-measurement -- "
                      "reporting energy_j=None for this call rather than guessing the "
                      "wrapped value.")
                delta_uj = None
            self.energy_j = None if delta_uj is None else delta_uj / 1e6
            self.energy_method = 'rapl'
        else:
            cpu_frac = self._proc.cpu_percent(interval=None) / 100.0
            self.energy_j = ASSUMED_CPU_TDP_WATTS * cpu_frac * self.latency_s
            self.energy_method = 'proxy_psutil'
        return False
