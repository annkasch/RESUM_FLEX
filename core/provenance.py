"""Which code and library versions wrote an output: the ``code`` block of run JSON files.

``code_provenance(source)`` returns::

    source        the source directory git was asked about (resum_flex's by default)
    commit        ``git rev-parse HEAD``, None if not a git checkout or git is missing
    branch        current branch, None on a detached HEAD
    dirty         True if tracked files differ from the commit (untracked files don't count)
    git_error     only if git failed: why commit/branch/dirty are None
    python        interpreter version
    packages      installed versions of ``packages`` (default: resum_flex and the libraries it
                  uses; None if absent)
    torch_cuda    CUDA version Torch was built with (None for a CPU build), only if Torch is
                  already imported
    command       the process's command line (``sys.argv``)
    modules       code loaded by name from a config (``data_provider``, ``event_source``; see
                  ``load_object``) outside resum_flex: module, source, git state of its checkout,
                  distribution and version

``save_config`` copies a stage's input YAML unchanged into its output directory
(``<stage>.yaml``), next to the resolved config.

``source`` and ``packages`` let an application describe its own repository the same way; the
git query runs in ``source``, not in the working directory.

numpy-free and torch-free, like ``core.compute``. Results are cached per source directory:
the first call of a process (at the start of a run) describes the code that run loaded.
"""

import importlib
import platform
import shutil
import subprocess
import sys
from functools import cache
from importlib import metadata
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
PACKAGES = (
    "resum-flex",
    "numpy",
    "scipy",
    "torch",
    "pydantic",
    "pyyaml",
    "matplotlib",
    "GPy",
    "emukit",
    "scikit-learn",
    "lightgbm",
)
GIT_TIMEOUT_SECONDS = 30


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout.strip()


def git_state(source: Path) -> dict:
    """Commit, branch and dirty flag of the git checkout containing ``source``."""
    try:
        commit = _git(source, "rev-parse", "HEAD")
        branch = _git(source, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git(source, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError) as error:
        stderr = getattr(error, "stderr", None)
        reason = stderr.strip() if isinstance(stderr, str) and stderr.strip() else str(error)
        return dict(commit=None, branch=None, dirty=None, git_error=reason)
    return dict(commit=commit, branch=None if branch == "HEAD" else branch, dirty=dirty)


def package_versions(names=PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@cache
def _cached(source: Path, packages: tuple[str, ...]) -> dict:
    return dict(
        source=str(source),
        **git_state(source),
        python=platform.python_version(),
        packages=package_versions(packages),
    )


_cached_git_state = cache(git_state)
# Modules loaded by load_object, in load order: name -> record (taken when first loaded).
_MODULES: dict[str, dict] = {}


def module_record(module) -> dict:
    """Where ``module`` was loaded from: source directory, its git state and distribution."""
    top = module.__name__.partition(".")[0]
    distributions = metadata.packages_distributions().get(top) or []
    distribution = distributions[0] if distributions else None
    try:
        version = None if distribution is None else metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None
    record = dict(module=module.__name__, distribution=distribution, version=version)
    path = getattr(module, "__file__", None)
    if path is None:  # namespace package: no file, no checkout to ask
        return dict(record, source=None, commit=None, branch=None, dirty=None)
    source = Path(path).resolve().parent
    return dict(record, source=str(source), **_cached_git_state(source))


def load_object(spec: str, what: str):
    """Attribute ``spec`` ('module:Name') named by config setting ``what``; records its module.

    Modules outside resum_flex appear in every later ``code`` block (``modules``), so a run
    records the application code (e.g. its dataset provider) along with resum_flex's.
    """
    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise ValueError(f"{what} must be 'module:Name', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import {what} {spec!r}: {exc}") from exc
    path = getattr(module, "__file__", None)
    inside = path is not None and Path(path).resolve().is_relative_to(SOURCE)
    if not inside and module.__name__ not in _MODULES:
        _MODULES[module.__name__] = module_record(module)
    return value


def code_provenance(source: str | Path = SOURCE, packages=PACKAGES) -> dict:
    """The ``code`` block: git state of ``source``, Python and package versions, loaded modules."""
    record = dict(_cached(Path(source).resolve(), tuple(packages)))
    record["packages"] = dict(record["packages"])
    torch = sys.modules.get("torch")
    record["torch_cuda"] = None if torch is None else torch.version.cuda
    record["command"] = list(sys.argv)
    record["modules"] = [dict(m) for m in _MODULES.values()]
    return record


def save_config(config: str | Path, directory: str | Path, stage: str) -> Path:
    """Copy the input YAML ``config`` unchanged to ``directory/<stage>.yaml``; never overwrite.

    Relative paths inside the copy still refer to the original file's directory; the stage's
    resolved config (e.g. ``experiment.json``) holds them resolved.
    """
    target = Path(directory) / f"{stage}.yaml"
    if target.exists():
        raise FileExistsError(f"{target} exists; refusing to overwrite a saved config")
    shutil.copyfile(config, target)
    return target
