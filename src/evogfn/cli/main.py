"""Command line entry point.

Hydra composes the run from configuration groups, so every component -- the
landscape, the environment, the reward, the policy, the tracker -- is selected
and overridden from the command line rather than edited in code::

    evogfn train
    evogfn train landscape=gb1 training.steps=5000
    evogfn train reward.beta=1.0 tracker=noop

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

from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis
from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import train_trajectory_balance
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.loop.campaign import Campaign
from evogfn.surrogate.proxy import ProxyLandscape
from evogfn.tracking.provenance import run_provenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evogfn.algorithms.base import Sampler
    from evogfn.loop.ledger import CampaignResult

#: Commands the entry point accepts.
COMMANDS = ("train", "campaign")

_USAGE = f"""usage: evogfn <command> [hydra overrides]

commands:
  {"  ".join(COMMANDS)}

examples:
  evogfn train
  evogfn train landscape=gb1 training.steps=5000
  evogfn campaign
  evogfn campaign sampler=genetic acquisition=ucb selector=diverse
  evogfn campaign campaign.rounds=8 campaign.batch_size=48
  evogfn <command> --help       show every configurable option
"""

#: Samplers the campaign command can drive, by name.
SAMPLERS = ("gflownet", "genetic", "hill-climb", "random")


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


@hydra.main(version_base=None, config_path="../configs", config_name="campaign")
def campaign(config: DictConfig) -> None:
    """Run a lab-in-the-loop campaign under a fixed oracle budget.

    Every sampler is driven by the same loop and charged for the same thing, so
    a difference between runs is a difference between methods rather than
    between harnesses. Only the measured batch is charged: a GFlowNet trains
    against a surrogate proxy, never the assay.

    Args:
        config: Composed Hydra configuration.

    Raises:
        ValueError: If ``sampler`` is not one this command can build.
    """
    torch.manual_seed(config.seed)

    landscape = hydra.utils.instantiate(config.landscape)
    env = hydra.utils.instantiate(
        config.env,
        parent=_starting_sequence(landscape),
        alphabet=landscape.alphabet,
        transitions=getattr(landscape, "transition_matrix", None),
        _convert_="object",
    )
    surrogate = hydra.utils.instantiate(
        config.surrogate,
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        seed=config.seed,
    )
    sampler = _build_sampler(config, env, surrogate, landscape)

    with hydra.utils.instantiate(config.tracker) as tracker:
        tracker.log_config(
            {
                "config": OmegaConf.to_container(config, resolve=True),
                **run_provenance(seed=config.seed),
            }
        )
        result = Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=surrogate,
            acquisition=hydra.utils.instantiate(config.acquisition),
            selector=hydra.utils.instantiate(config.selector),
            rounds=config.campaign.rounds,
            batch_size=config.campaign.batch_size,
            pool_size=config.campaign.pool_size,
            skip_measured=config.campaign.skip_measured,
            tracker=tracker,
        ).run()
        tracker.log_metrics(result.summary(), step=len(result.rounds))
    _report_campaign(result)


def _build_sampler(
    config: DictConfig, env: object, surrogate: object, landscape: object
) -> Sampler:
    """Construct the sampler named in the configuration.

    A GFlowNet is handed a proxy over the *same* surrogate instance the campaign
    refits, so it trains against each round's model without the loop needing to
    know which samplers care. The classical baselines get the same proxy access,
    because comparing a method that optimises the model against one that only
    meets it as a filter is not a comparison of methods.
    """
    name = str(config.sampler)
    if name == "gflownet":
        policy = hydra.utils.instantiate(
            config.policy,
            n_tokens=landscape.alphabet.size,  # type: ignore[attr-defined]
            sequence_length=landscape.sequence_length,  # type: ignore[attr-defined]
            n_actions=env.n_actions,  # type: ignore[attr-defined]
        )
        return GFlowNetSampler(
            env,  # type: ignore[arg-type]
            policy,
            proxy=ProxyLandscape(
                surrogate,  # type: ignore[arg-type]
                alphabet=landscape.alphabet,  # type: ignore[attr-defined]
                sequence_length=landscape.sequence_length,  # type: ignore[attr-defined]
            ),
            reward=hydra.utils.instantiate(config.reward),
            config=hydra.utils.instantiate(config.training),
            objective=hydra.utils.instantiate(config.objective),
            seed=config.seed,
        )

    builders = {
        "genetic": GeneticAlgorithm,
        "hill-climb": HillClimbing,
        "random": RandomMutagenesis,
    }
    if name not in builders:
        raise ValueError(f"unknown sampler {name!r}; expected one of {SAMPLERS}")
    inner = builders[name](env, seed=config.seed)
    return ProxyOptimising(
        inner,
        proxy=ProxyLandscape(
            surrogate,  # type: ignore[arg-type]
            alphabet=landscape.alphabet,  # type: ignore[attr-defined]
            sequence_length=landscape.sequence_length,  # type: ignore[attr-defined]
        ),
    )


def _report_campaign(result: CampaignResult) -> None:
    """Print the ledger, so a run is readable without opening a tracker."""
    print(f"\n{result.sampler}: {result.oracle_calls} oracle calls")
    for record in result.rounds:
        print(
            f"  round {record.index}: measured {record.evaluated:>4}  "
            f"best {record.best_so_far:>8.4f}  "
            f"feasible {record.feasible_fraction:.3f}  "
            f"diversity {record.batch_diversity:.2f}"
        )
    if (regret := result.simple_regret) is not None:
        print(f"  simple regret {regret:.4f}")


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
    {"train": train, "campaign": campaign}[command]()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
