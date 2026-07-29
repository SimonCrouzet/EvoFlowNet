"""Recording run metrics, configuration and artefacts.

Behind an interface rather than a direct vendor dependency, so tests and CI run
with no account, no API key and no network -- and so a benchmark does not
require access to one company's service to reproduce.
"""

from evogfn.tracking.base import MultiTracker, NoOpTracker, Tracker
from evogfn.tracking.console import ConsoleTracker
from evogfn.tracking.provenance import git_provenance, run_provenance

__all__ = [
    "ConsoleTracker",
    "MultiTracker",
    "NoOpTracker",
    "Tracker",
    "git_provenance",
    "run_provenance",
]
