"""Learned stand-ins for an expensive oracle."""

from evoflownet.surrogate.base import Surrogate
from evoflownet.surrogate.ensemble import DEFAULT_MEMBERS, DeepEnsemble
from evoflownet.surrogate.proxy import ProxyLandscape

__all__ = ["DEFAULT_MEMBERS", "DeepEnsemble", "ProxyLandscape", "Surrogate"]
