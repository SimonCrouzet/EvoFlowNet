"""Command line entry point.

Hydra composes the run from configuration groups, so every component -- the
landscape, the environment, the reward, the policy, the tracker -- is selected
and overridden from the command line rather than edited in code::

    evoflownet train
    evoflownet train landscape=gb1 training.steps=5000
    evoflownet train reward.beta=1.0 tracker=noop

Hydra has no notion of subcommands, so the first argument is taken as one and
removed before Hydra sees the rest. That keeps room for ``benchmark`` and
``campaign`` alongside ``train`` without a second entry point.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from evoflownet.algorithms.gflownet.training import train_trajectory_balance
from evoflownet.tracking.provenance import run_provenance

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Commands the entry point accepts. Only ``train`` is implemented so far.
COMMANDS = ("train",)

_USAGE = f"""usage: evoflownet <command> [hydra overrides]

commands:
  {"  ".join(COMMANDS)}

examples:
  evoflownet train
  evoflownet train landscape=gb1 training.steps=5000
  evoflownet train reward.beta=1.0 tracker=noop
  evoflownet train --help          show every configurable option
"""


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(config: DictConfig) -> None:
    """Train a GFlowNet to sample proportionally to reward.

    Args:
        config: Composed Hydra configuration.
    """
    torch.manual_seed(config.seed)

    landscape = hydra.utils.instantiate(config.landscape)
    # The environment starts from a parent and writes in the landscape's
    # alphabet, so both come from it rather than being configured twice and
    # risking disagreement.
    # _convert_="object" stops Hydra structuring the Alphabet dataclass into a
    # config node, which would strip the properties the environment reads.
    env = hydra.utils.instantiate(
        config.env,
        parent=_starting_sequence(landscape),
        alphabet=landscape.alphabet,
        _convert_="object",
    )
    policy = hydra.utils.instantiate(
        config.policy,
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        n_actions=env.n_actions,
    )
    reward = hydra.utils.instantiate(config.reward)
    training = hydra.utils.instantiate(config.training)
    objective = hydra.utils.instantiate(config.objective)

    with hydra.utils.instantiate(config.tracker) as tracker:
        tracker.log_config(
            {
                "config": OmegaConf.to_container(config, resolve=True),
                **run_provenance(seed=config.seed),
            }
        )
        result = train_trajectory_balance(
            env,
            policy,
            landscape,
            reward,
            training,
            objective=objective,
            tracker=tracker,
        )
        final = {
            "final_loss": result.losses[-1],
            "oracle_calls": float(result.oracle_calls),
        }
        if objective.uses_log_z:
            final["final_log_z"] = result.final_log_z
        tracker.log_metrics(final, step=training.steps)


def _starting_sequence(landscape: object) -> np.ndarray:
    """The parent every trajectory starts from.

    Uses the landscape's wild type where it has one, since directed evolution
    starts from a real sequence rather than an arbitrary point. Falls back to
    the first token repeated, which is well defined for any alphabet.

    Args:
        landscape: The landscape being optimised against.

    Returns:
        A ``(sequence_length,)`` array of token indices.
    """
    wild_type = getattr(landscape, "wild_type", None)
    if wild_type is not None:
        return np.asarray(wild_type, dtype=np.int32)
    length = landscape.sequence_length  # type: ignore[attr-defined]
    return np.zeros(length, dtype=np.int32)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a subcommand, then hand the rest of the arguments to Hydra.

    Args:
        argv: Arguments excluding the program name. Defaults to ``sys.argv``.

    Returns:
        A process exit code.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_USAGE)
        return 0
    command, rest = arguments[0], arguments[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n\n{_USAGE}", file=sys.stderr)
        return 2

    # Hydra reads sys.argv directly, so the subcommand has to be removed.
    sys.argv = [sys.argv[0], *rest]
    train()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
