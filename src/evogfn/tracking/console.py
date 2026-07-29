"""A tracker that prints, for runs with nowhere else to send their metrics.

The default, so that the package is usable and CI is runnable with no account,
no API key and no network. It is deliberately quiet: printing every step of a
long run buries the information rather than conveying it, so metrics are emitted
at an interval.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, TextIO

from evogfn.tracking.base import Tracker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ConsoleTracker(Tracker):
    """Writes configuration, metrics and artefact names to a stream.

    Args:
        stream: Where to write. Defaults to stderr, so that metrics do not
            contaminate anything a caller is piping from stdout.
        every: Emit metrics on steps divisible by this. Step 0 and the final
            call via :meth:`finish` are always emitted, so a short run is never
            silent.
        precision: Decimal places for metric values.

    Raises:
        ValueError: If ``every`` is not positive.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        every: int = 100,
        precision: int = 4,
    ) -> None:
        """Configure the output stream and interval."""
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self._stream = stream if stream is not None else sys.stderr
        self._every = every
        self._precision = precision
        self._last: tuple[int, Mapping[str, float]] | None = None
        self._emitted_last = True

    def log_config(self, config: Mapping[str, object]) -> None:
        """Print the configuration as indented JSON.

        Args:
            config: Resolved configuration.
        """
        # default=str so an unserialisable value is reported rather than
        # crashing a run at its very first call.
        rendered = json.dumps(config, indent=2, sort_keys=True, default=str)
        print(f"config:\n{rendered}", file=self._stream, flush=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Print metrics, subject to the interval.

        Args:
            metrics: Metric names to values.
            step: The training step.
        """
        self._last = (step, dict(metrics))
        if step % self._every == 0:
            self._emit(step, metrics)
            self._emitted_last = True
        else:
            self._emitted_last = False

    def log_artifact(self, path: Path, *, name: str) -> None:
        """Print the artefact's name and location.

        Args:
            path: The file produced.
            name: A name for it.
        """
        print(f"artifact {name}: {path}", file=self._stream, flush=True)

    def finish(self) -> None:
        """Emit the final metrics if the interval would otherwise have skipped them.

        Without this, the last and most interesting step of a run is usually the
        one not printed.
        """
        if self._last is not None and not self._emitted_last:
            self._emit(*self._last)
            self._emitted_last = True

    def _emit(self, step: int, metrics: Mapping[str, float]) -> None:
        """Write one metrics line."""
        body = "  ".join(f"{k}={v:.{self._precision}f}" for k, v in sorted(metrics.items()))
        print(f"step {step:>7}  {body}", file=self._stream, flush=True)
