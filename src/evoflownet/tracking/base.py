"""Where run metrics and artefacts go.

Tracking is an interface rather than a direct Weights & Biases dependency for
two reasons. Tests and CI must run with no account, no API key and no network,
which a hard dependency makes impossible. And a benchmark that can only be
reproduced by someone with access to one vendor's service is not reproducible.

The default is :class:`~evoflownet.tracking.console.ConsoleTracker`, which needs
nothing. W&B is an opt-in adapter behind the ``wandb`` extra.

Run provenance
--------------

:meth:`Tracker.log_config` exists because a metric without the configuration
that produced it is not a result. Implementations are expected to record the
resolved config, the git commit, and whether the working tree was dirty --
the last being the one people omit and the one that makes a number
irreproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import TracebackType


class Tracker(ABC):
    """Records metrics, configuration and artefacts from a run.

    Usable as a context manager, so a run is closed even when training raises --
    otherwise a crashed run leaves no record of how far it got, which is exactly
    when the record is wanted.
    """

    @abstractmethod
    def log_config(self, config: Mapping[str, object]) -> None:
        """Record the configuration a run was launched with.

        Args:
            config: Resolved configuration, already flattened or nested.
        """

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Record scalar metrics at a point in training.

        Args:
            metrics: Metric names to values.
            step: The training step these belong to.
        """

    @abstractmethod
    def log_artifact(self, path: Path, *, name: str) -> None:
        """Record a file produced by the run.

        Args:
            path: The file to record.
            name: A name for it.
        """

    def finish(self) -> None:  # noqa: B027 - a default no-op, not an abstract method
        """Flush and close. Implementations that need no teardown may ignore it."""

    def __enter__(self) -> Tracker:
        """Enter a run context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the run, including when the body raised."""
        self.finish()


class NoOpTracker(Tracker):
    """Discards everything.

    The default in tests, where the assertion is about what was computed rather
    than about what was reported.
    """

    def log_config(self, config: Mapping[str, object]) -> None:
        """Discard the configuration."""

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Discard the metrics."""

    def log_artifact(self, path: Path, *, name: str) -> None:
        """Discard the artefact."""


class MultiTracker(Tracker):
    """Forwards to several trackers at once.

    Lets a run print to the console *and* record to a service without either
    knowing about the other.

    Args:
        trackers: The trackers to forward to, in order.
    """

    def __init__(self, *trackers: Tracker) -> None:
        """Store the trackers to forward to."""
        self._trackers = trackers

    @property
    def trackers(self) -> tuple[Tracker, ...]:
        """The trackers being forwarded to."""
        return self._trackers

    def log_config(self, config: Mapping[str, object]) -> None:
        """Forward the configuration to every tracker."""
        for tracker in self._trackers:
            tracker.log_config(config)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Forward the metrics to every tracker."""
        for tracker in self._trackers:
            tracker.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path, *, name: str) -> None:
        """Forward the artefact to every tracker."""
        for tracker in self._trackers:
            tracker.log_artifact(path, name=name)

    def finish(self) -> None:
        """Close every tracker, even if one of them raises.

        A tracker that fails to close must not prevent the others from closing:
        losing one destination's record is a nuisance, losing all of them
        because the first failed is a lost run.
        """
        errors: list[Exception] = []
        for tracker in self._trackers:
            try:
                tracker.finish()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("one or more trackers failed to close", errors)
