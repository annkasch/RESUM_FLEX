"""Which code and library versions wrote an output: the ``code`` block of run JSON files.

``code_provenance(source)`` returns::

    source        the source directory git was asked about (resum_flex's by default)
    commit        ``git rev-parse HEAD``, None if not a git checkout or git is missing
    branch        current branch, None on a detached HEAD
    dirty         True if tracked files differ from the commit (untracked files don't count)
    git_error     only if git failed: why commit/branch/dirty are None
    python        interpreter version
    packages      installed versions of resum_flex and the libraries it uses (None if absent)
    torch_cuda    CUDA version Torch was built with (None for a CPU build), only if Torch is
                  already imported
    command       the process's command line (``sys.argv``)

``save_config`` copies a stage's input YAML unchanged into its output directory
(``<stage>.yaml``), next to the resolved config.

resum_flex records its own code only. ``source`` lets an application describe its own
repository the same way; the git query runs in ``source``, not in the working directory.

numpy-free and torch-free, like ``core.compute``. Results are cached per source directory:
the first call of a process (at the start of a run) describes the code that run loaded.
"""

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
def _cached(source: Path) -> dict:
    return dict(
        source=str(source),
        **git_state(source),
        python=platform.python_version(),
        packages=package_versions(),
    )


def code_provenance(source: str | Path = SOURCE) -> dict:
    """The ``code`` block: git state of ``source`` plus Python and package versions."""
    record = dict(_cached(Path(source).resolve()))
    record["packages"] = dict(record["packages"])
    torch = sys.modules.get("torch")
    record["torch_cuda"] = None if torch is None else torch.version.cuda
    record["command"] = list(sys.argv)
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
