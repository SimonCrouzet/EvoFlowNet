"""Classical baselines, driven through the same interface as the GFlowNet."""

from evogfn.algorithms.baselines.annealing import SimulatedAnnealing
from evogfn.algorithms.baselines.cmaes import CMAES
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.mlde import (
    DEFAULT_TRAINING_SIZE,
    MLDE,
    PUBLISHED_BATCH_SIZE,
    PUBLISHED_BUDGET,
    PUBLISHED_CV_FOLDS,
    PUBLISHED_MODELS_AVERAGED,
    PUBLISHED_TRAINING_SIZE,
)
from evogfn.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis

__all__ = [
    "CMAES",
    "DEFAULT_TRAINING_SIZE",
    "MLDE",
    "PUBLISHED_BATCH_SIZE",
    "PUBLISHED_BUDGET",
    "PUBLISHED_CV_FOLDS",
    "PUBLISHED_MODELS_AVERAGED",
    "PUBLISHED_TRAINING_SIZE",
    "GeneticAlgorithm",
    "HillClimbing",
    "RandomMutagenesis",
    "SimulatedAnnealing",
]
