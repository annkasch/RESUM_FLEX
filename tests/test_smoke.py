"""End-to-end CLI pipeline check (scripts/smoke.py); skip in quick runs with -m 'not smoke'."""

import pytest

pytest.importorskip("GPy")


@pytest.mark.smoke
def test_every_cli_stage_runs_on_synthetic_data(tmp_path):
    from scripts.smoke import run

    timings = dict(run(tmp_path / "smoke"))
    assert {"train", "stream", "mfgp", "binomial"} <= timings.keys()
    assert sum(name.startswith("predict") for name in timings) == 3
