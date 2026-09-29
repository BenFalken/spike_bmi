"""
EnergyMeter: wall time and CPU energy of a block of code.

    with EnergyMeter() as meter:
        run_workload()
    meter.latency_s, meter.energy_j, meter.energy_method   # 'rapl' or 'proxy_psutil'

Energy comes from the Intel RAPL counters of the CPU package domains
(/sys/class/powercap/intel-rapl/intel-rapl:N with name 'package-N'; platform
domains such as 'psys' are excluded). Where RAPL is not readable it falls back
to a proxy, elapsed time x process CPU utilization x an assumed 65 W TDP, which
is only meaningful for comparing workloads with each other.

Importing this module also pins the OpenMP/BLAS thread counts to the Slurm
allocation (or ENERGY_METER_NUM_THREADS). Those libraries read the setting
when they are first imported, so import this module before numpy, torch or
TensorFlow, and call pin_torch_threads() after importing torch.
"""

import glob
import os
import time

import psutil

ASSUMED_CPU_TDP_WATTS = 65.0


def pin_math_library_threads():
    """Set the OpenMP/BLAS thread-count variables; returns the count or None."""
    n_threads = (os.environ.get("ENERGY_METER_NUM_THREADS")
                 or os.environ.get("TEST_ALL_DECODERS_NUM_THREADS")
                 or os.environ.get("SLURM_CPUS_PER_TASK"))
    if not n_threads:
        print("WARNING: no ENERGY_METER_NUM_THREADS or SLURM_CPUS_PER_TASK; math libraries "
              "will use every core, which makes latency and energy noisy on a shared node.")
        return None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n_threads
    print(f"Pinning math library thread counts to {n_threads}")
    return int(n_threads)


NUM_THREADS = pin_math_library_threads()


def pin_torch_threads():
    """Pin PyTorch's intra-op thread pool as well (call after importing torch)."""
    if NUM_THREADS is not None:
        import torch
        torch.set_num_threads(NUM_THREADS)


def _rapl_package_paths():
    paths = []
    for path in sorted(glob.glob('/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj')):
        try:
            with open(os.path.join(os.path.dirname(path), 'name')) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name.startswith('package-') and name[len('package-'):].isdigit():
            paths.append(path)
    return paths


_RAPL_PATHS = _rapl_package_paths()


def _read_rapl_uj():
    """Summed package energy in microjoules, or None if unreadable."""
    if not _RAPL_PATHS:
        return None
    try:
        total = 0
        for path in _RAPL_PATHS:
            with open(path) as f:
                total += int(f.read().strip())
        return total
    except (OSError, ValueError):
        return None


RAPL_AVAILABLE = _read_rapl_uj() is not None
print(f"Energy measurement: {'Intel RAPL' if RAPL_AVAILABLE else 'CPU-utilization proxy (RAPL not readable)'}")


class EnergyMeter:
    """Context manager setting .latency_s, .energy_j and .energy_method on
    exit. energy_j is None if a RAPL counter wrapped during the block."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        if RAPL_AVAILABLE:
            self._e0 = _read_rapl_uj()
        else:
            self._proc = psutil.Process()
            self._proc.cpu_percent(interval=None)   # first call only sets the baseline
        return self

    def __exit__(self, *exc_info):
        self.latency_s = time.perf_counter() - self._t0
        if RAPL_AVAILABLE:
            e1 = _read_rapl_uj()
            delta = None if e1 is None else e1 - self._e0
            if delta is not None and delta < 0:
                print("  WARNING: RAPL counter wrapped during the measurement; energy_j=None")
                delta = None
            self.energy_j = None if delta is None else delta / 1e6
            self.energy_method = 'rapl'
        else:
            cpu_fraction = self._proc.cpu_percent(interval=None) / 100.0
            self.energy_j = ASSUMED_CPU_TDP_WATTS * cpu_fraction * self.latency_s
            self.energy_method = 'proxy_psutil'
        return False
