"""Code provenance: git state of a source directory, package versions, compute.json block."""

import json
import subprocess

import pytest

from core.compute import ComputeLog
from core.provenance import PACKAGES, SOURCE, code_provenance, git_state, package_versions


def _git(path, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(path), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "work")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_clean_then_dirty_checkout(repo):
    state = git_state(repo)
    assert len(state["commit"]) == 40 and state["branch"] == "work" and state["dirty"] is False
    (repo / "scratch.txt").write_text("untracked\n")  # untracked files don't count
    assert git_state(repo)["dirty"] is False
    (repo / "a.txt").write_text("two\n")
    assert git_state(repo)["dirty"] is True


def test_detached_head_has_no_branch(repo):
    _git(repo, "checkout", "-q", "--detach")
    assert git_state(repo)["branch"] is None


def test_not_a_checkout_records_why(tmp_path):
    state = git_state(tmp_path)
    assert state["commit"] is state["branch"] is state["dirty"] is None
    assert state["git_error"]


def test_package_versions_mark_missing_packages():
    versions = package_versions(("numpy", "no-such-package-xyz"))
    assert versions["numpy"] and versions["no-such-package-xyz"] is None


def test_code_block_describes_resum_flex(repo):
    code = code_provenance()
    assert code["source"] == str(SOURCE)
    assert set(code["packages"]) == set(PACKAGES) and code["python"]
    # An application can describe its own repository the same way.
    assert code_provenance(repo)["branch"] == "work"


def test_compute_json_carries_the_code_block(tmp_path):
    summary = json.loads(ComputeLog().write(tmp_path).read_text())
    assert summary["code"]["source"] == str(SOURCE)
    assert "commit" in summary["code"] and "torch_cuda" in summary["code"]
