"""Weights & Biases adapter, behind the ``wandb`` extra.

Imported lazily so that the package, its tests and CI never require the
dependency, an account, or a network connection. Selecting this tracker without
having installed the extra fails with an instruction rather than an
``ImportError`` from three frames down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evoflownet.tracking.base import Tracker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class WandbTracker(Tracker):
    """Records a run to Weights & Biases.

    Args:
        project: W&B project name.
        name: Run name. Left to W&B to generate when omitted.
        entity: W&B entity (team or user).
        **options: Passed through to ``wandb.init``.

    Raises:
        ImportError: If the ``wandb`` extra is not installed.
    """

    def __init__(
        self,
        *,
        project: str = "evoflownet",
        name: str | None = None,
        entity: str | None = None,
        **options: Any,  # noqa: ANN401
    ) -> None:
        """Start a W&B run."""
        try:
            import wandb  # noqa: PLC0415 - deliberately lazy
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the wandb tracker needs the optional dependency: "
                "install with `uv sync --extra wandb`, or select a different tracker"
            ) from error

        self._wandb = wandb
        self._run = wandb.init(project=project, name=name, entity=entity, **options)

    @property
    def run(self) -> Any:  # noqa: ANN401 - the vendor's own run object
        """The underlying W&B run."""
        return self._run

    def log_config(self, config: Mapping[str, object]) -> None:
        """Record the configuration on the run.

        Args:
            config: Resolved configuration.
        """
        self._run.config.update(dict(config), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Record metrics at a step.

        Args:
            metrics: Metric names to values.
            step: The training step.
        """
        self._run.log(dict(metrics), step=step)

    def log_artifact(self, path: Path, *, name: str) -> None:
        """Upload a file as a W&B artifact.

        Args:
            path: The file to upload.
            name: Artifact name.
        """
        artifact = self._wandb.Artifact(name=name, type="file")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Close the W&B run."""
        self._run.finish()
