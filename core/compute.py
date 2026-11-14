"""Compute accounting per pipeline stage: wall time, CPU time and allocated cores.

Each run writes ``compute.json``::

    stages       one record per stage (data preparation, training, fit, evaluation, ...)
    total        the whole run, from the start of the log to the summary
    environment  host, Slurm job, allocated CPUs/cores, thread settings
    code         resum_flex commit, dirty flag, Python and package versions (core.provenance),
                 taken when the log is created

Per record:

- ``wall_seconds``: clock time.
- ``cpu_seconds``: CPU time actually used, summed over all threads (and waited-for child
  processes), from ``resource.getrusage``.
- ``allocated_cores``: physical cores this process may run on (its CPU affinity, which is
  the Slurm allocation inside a job). ``core_hours = wall_seconds * allocated_cores / 3600``
  is what the allocation costs, used or not.
- ``cpu_utilization = cpu_seconds / (wall_seconds * allocated_cores)``: the share of the
  allocated cores that was busy. Logical CPUs (hyperthreads) count towards their core.

numpy-free and torch-free: torch threads are read only if torch is already imported.
"""

import json
import os
import resource
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from core.provenance import code_provenance

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def allocated_cpus() -> list[int]:
    """Logical CPU ids this process may run on."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return list(range(os.cpu_count() or 1))


def physical_cores(cpus: list[int], sysfs: Path = Path("/sys/devices/system/cpu")) -> int:
    """Number of distinct physical cores among logical ``cpus`` (hyperthreads merged).

    Falls back to the number of logical CPUs when the topology is not readable.
    """
    cores = set()
    try:
        for cpu in cpus:
            topology = sysfs / f"cpu{cpu}" / "topology"
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
            cores.add((package, core))
    except OSError:
        return len(cpus)
    return len(cores) or len(cpus)


def _cpu_seconds() -> float:
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def _torch_threads() -> int | None:
    torch = sys.modules.get("torch")
    return None if torch is None else int(torch.get_num_threads())


def environment() -> dict:
    """Host, Slurm job and thread settings of this process."""
    cpus = allocated_cpus()
    return dict(
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_cpus_on_node=os.environ.get("SLURM_CPUS_ON_NODE"),
        allocated_cpus=len(cpus),
        allocated_cores=physical_cores(cpus),
        gpus=os.environ.get("SLURM_GPUS_ON_NODE", "0"),
        thread_environment={name: os.environ.get(name) for name in THREAD_VARIABLES},
    )


def _record(name, wall, cpu, cores, torch_threads):
    return dict(
        name=name,
        wall_seconds=wall,
        cpu_seconds=cpu,
        allocated_cores=cores,
        core_hours=wall * cores / 3600,
        cpu_utilization=cpu / (wall * cores) if wall > 0 and cores else None,
        torch_threads=torch_threads,
    )


class ComputeLog:
    """Stage records of one run; the total runs from construction to :meth:`summary`.

    Stages are measured either as a block (``with log.stage(name):``) or as laps
    (``log.lap(name)`` records everything since the previous lap, stage or construction).
    """

    def __init__(self):
        self.stages: list[dict] = []
        self._wall0, self._cpu0 = time.perf_counter(), _cpu_seconds()
        self._mark = (self._wall0, self._cpu0)
        self._cores = physical_cores(allocated_cpus())
        self.code = code_provenance()

    def lap(self, name: str) -> None:
        """Record the time since the previous mark as stage ``name``."""
        wall, cpu = time.perf_counter(), _cpu_seconds()
        self.stages.append(
            _record(name, wall - self._mark[0], cpu - self._mark[1], self._cores, _torch_threads())
        )
        self._mark = (wall, cpu)

    @contextmanager
    def stage(self, name: str):
        """Measure the enclosed block as stage ``name`` (recorded also if it raises)."""
        wall0, cpu0, threads = time.perf_counter(), _cpu_seconds(), _torch_threads()
        try:
            yield
        finally:
            end = (time.perf_counter(), _cpu_seconds())
            self.stages.append(_record(name, end[0] - wall0, end[1] - cpu0, self._cores, threads))
            self._mark = end

    def summary(self) -> dict:
        wall, cpu = time.perf_counter() - self._wall0, _cpu_seconds() - self._cpu0
        return dict(
            stages=list(self.stages),
            total=_record("total", wall, cpu, self._cores, _torch_threads()),
            environment=environment(),
            # resum_flex's state is the one taken at construction (cached); modules loaded
            # since (e.g. the dataset provider) are added now.
            code=code_provenance(),
        )

    def write(self, directory) -> Path:
        """Write ``compute.json`` into ``directory`` and return its path."""
        path = Path(directory) / "compute.json"
        path.write_text(json.dumps(self.summary(), indent=2, allow_nan=False))
        return path
