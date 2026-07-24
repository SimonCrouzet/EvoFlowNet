"""Fitness landscapes: the functions being optimised against."""

from evoflownet.landscapes.base import MAX_ENUMERABLE_SIZE, FitnessLandscape
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.landscapes.wrappers import Budgeted, BudgetExhaustedError, Cached, Noisy

__all__ = [
    "MAX_ENUMERABLE_SIZE",
    "BudgetExhaustedError",
    "Budgeted",
    "Cached",
    "EhrlichLandscape",
    "FitnessLandscape",
    "Noisy",
]
