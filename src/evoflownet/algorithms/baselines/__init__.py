"""Classical baselines, driven through the same interface as the GFlowNet."""

from evoflownet.algorithms.baselines.annealing import SimulatedAnnealing
from evoflownet.algorithms.baselines.cmaes import CMAES
from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
from evoflownet.algorithms.baselines.mlde import (
    DEFAULT_TRAINING_SIZE,
    MLDE,
    PUBLISHED_BATCH_SIZE,
    PUBLISHED_BUDGET,
    PUBLISHED_CV_FOLDS,
    PUBLISHED_MODELS_AVERAGED,
    PUBLISHED_TRAINING_SIZE,
)
from evoflownet.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis

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
