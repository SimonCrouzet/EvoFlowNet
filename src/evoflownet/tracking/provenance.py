"""What a run has to record about itself to be reproducible.

A metric with no record of the code that produced it is not a result. The three
things people omit, in increasing order of how much damage the omission does:
the package version, the git commit, and whether the working tree was dirty.

The last is the important one. A commit hash implies "this code produced this
number", and if there were uncommitted changes that implication is false, with
nothing in the record to say so.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from evoflownet import __version__


def _git(*args: str) -> str | None:
    """Run a git command, returning ``None`` if it fails or git is absent."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            ["git", *args],  # noqa: S607 - resolved from PATH deliberately
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Describe the working tree a run was launched from.

    Returns:
        A mapping with ``commit``, ``branch`` and ``dirty``. Values are ``None``
        when the run is not inside a git repository, which is itself worth
        recording rather than hiding.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A commit hash alone claims this code produced this number. With
        # uncommitted changes that claim is false, so it is recorded explicitly.
        "dirty": bool(status),
    }


def run_provenance(**extra: Any) -> dict[str, Any]:  # noqa: ANN401 - forwarded verbatim
    """Everything needed to reproduce a run, apart from its configuration.

    Args:
        **extra: Additional fields to record, such as the seed.

    Returns:
        A mapping suitable for :meth:`~evoflownet.tracking.base.Tracker.log_config`.
    """
    return {
        "evoflownet_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_provenance(),
        **extra,
    }
