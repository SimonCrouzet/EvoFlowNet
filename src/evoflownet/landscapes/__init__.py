"""Fitness landscapes: the functions being optimised against."""

from evoflownet.landscapes.base import MAX_ENUMERABLE_SIZE, FitnessLandscape
from evoflownet.landscapes.ehrlich import EhrlichLandscape

__all__ = ["MAX_ENUMERABLE_SIZE", "EhrlichLandscape", "FitnessLandscape"]
