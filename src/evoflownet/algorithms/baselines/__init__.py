"""Classical baselines, driven through the same interface as the GFlowNet."""

from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
from evoflownet.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis

__all__ = ["GeneticAlgorithm", "HillClimbing", "RandomMutagenesis"]
