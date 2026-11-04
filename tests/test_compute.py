"""Compute accounting: stage records, laps, totals and physical-core counting."""

import json
import time

import pytest

from core.compute import ComputeLog, physical_cores

STAGE_KEYS = {
    "name",
    "wall_seconds",
    "cpu_seconds",
    "allocated_cores",
    "core_hours",
    "cpu_utilization",
    "torch_threads",
}


def _busy(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_stages_laps_and_total(tmp_path):
    log = ComputeLog()
    with log.stage("work"):
        _busy(0.05)
    time.sleep(0.05)  # idle: wall time without CPU time
    log.lap("idle")
    path = log.write(tmp_path)
    summary = json.loads(path.read_text())
    work, idle = summary["stages"]
    assert [work["name"], idle["name"]] == ["work", "idle"]
    assert set(work) == STAGE_KEYS == set(summary["total"])
    assert work["wall_seconds"] >= 0.05 and work["cpu_seconds"] > 0.03
    assert idle["wall_seconds"] >= 0.05 and idle["cpu_seconds"] < 0.03
    cores = work["allocated_cores"]
    assert work["core_hours"] == pytest.approx(work["wall_seconds"] * cores / 3600)
    total = summary["total"]
    assert total["wall_seconds"] >= work["wall_seconds"] + idle["wall_seconds"]
    assert summary["environment"]["allocated_cores"] == cores >= 1
    assert set(summary["environment"]["thread_environment"]) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }


def test_stage_is_recorded_when_it_raises():
    log = ComputeLog()
    with pytest.raises(RuntimeError), log.stage("failing"):
        raise RuntimeError
    assert [s["name"] for s in log.stages] == ["failing"]


def test_physical_cores_merge_hyperthreads(tmp_path):
    # cpu0/cpu2 and cpu1/cpu3 are hyperthread pairs on one package.
    for cpu, core in [(0, 0), (1, 1), (2, 0), (3, 1)]:
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text(f"{core}\n")
        (topology / "physical_package_id").write_text("0\n")
    assert physical_cores([0, 1, 2, 3], sysfs=tmp_path) == 2
    assert physical_cores([0, 2], sysfs=tmp_path) == 1
    # Unreadable topology: count logical CPUs.
    assert physical_cores([0, 1, 7], sysfs=tmp_path / "missing") == 3
