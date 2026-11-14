"""Code provenance: git state of a source directory, package versions, compute.json block."""

import json
import subprocess

import pytest

from core.compute import ComputeLog
from core.provenance import (
    PACKAGES,
    SOURCE,
    code_provenance,
    git_state,
    package_versions,
    save_config,
)


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


def test_code_block_records_the_command_line(monkeypatch):
    monkeypatch.setattr("sys.argv", ["resum", "train", "run.yaml"])
    assert code_provenance()["command"] == ["resum", "train", "run.yaml"]


def test_save_config_copies_verbatim_and_never_overwrites(tmp_path):
    source = tmp_path / "in.yaml"
    source.write_text("# my comment\nseed: 3  # chosen\noutput_directory: ../runs/a\n")
    output = tmp_path / "run"
    output.mkdir()
    target = save_config(source, output, "train")
    assert target == output / "train.yaml" and target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        save_config(source, output, "train")


def _external_module(repo, name):
    """A module file inside the git checkout ``repo``, importable as ``name``."""
    (repo / f"{name}.py").write_text(
        "class Factory:\n    def __init__(self, *args):\n        pass\n"
    )
    _git(repo, "add", f"{name}.py")
    _git(repo, "commit", "-q", "-m", "module")


def test_loaded_external_module_is_in_later_code_blocks(repo, monkeypatch):
    from core.provenance import load_object

    monkeypatch.setattr("core.provenance._MODULES", {})
    _external_module(repo, "provtest_app_provider")
    monkeypatch.syspath_prepend(str(repo))
    assert code_provenance()["modules"] == []
    load_object("provtest_app_provider:Factory", "data_provider")
    load_object("provtest_app_provider:Factory", "data_provider")  # recorded once
    (module,) = code_provenance()["modules"]
    assert module["module"] == "provtest_app_provider" and module["source"] == str(repo)
    assert module["branch"] == "work" and module["dirty"] is False and len(module["commit"]) == 40
    assert module["distribution"] is None  # not installed: the checkout identifies it


def test_resum_flex_modules_are_not_listed_twice(monkeypatch):
    from core.provenance import load_object

    monkeypatch.setattr("core.provenance._MODULES", {})
    load_object("core.surrogates.datasets:NpzBatches", "data_provider")
    assert code_provenance()["modules"] == []


def test_installed_module_records_its_distribution(monkeypatch):
    import yaml

    from core.provenance import module_record

    record = module_record(yaml)
    assert record["distribution"] == "PyYAML" and record["version"]


def test_load_object_rejects_bad_specs():
    from core.provenance import load_object

    with pytest.raises(ValueError, match="module:Name"):
        load_object("no_colon", "data_provider")
    with pytest.raises(ValueError, match="Cannot import data_provider"):
        load_object("no_such_module_xyz:Thing", "data_provider")


def test_application_chooses_its_packages(repo):
    code = code_provenance(repo, packages=("numpy", "no-such-package-xyz"))
    assert set(code["packages"]) == {"numpy", "no-such-package-xyz"}
    assert code["packages"]["no-such-package-xyz"] is None


def test_compute_json_lists_modules_loaded_after_construction(repo, tmp_path, monkeypatch):
    from core.provenance import load_object

    monkeypatch.setattr("core.provenance._MODULES", {})
    _external_module(repo, "provtest_late_provider")
    monkeypatch.syspath_prepend(str(repo))
    log = ComputeLog()
    load_object("provtest_late_provider:Factory", "data_provider")
    out = tmp_path / "run"
    out.mkdir()
    summary = json.loads(log.write(out).read_text())
    assert [m["module"] for m in summary["code"]["modules"]] == ["provtest_late_provider"]
