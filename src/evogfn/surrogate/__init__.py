"""Learned stand-ins for an expensive oracle."""

from evogfn.surrogate.base import Surrogate
from evogfn.surrogate.ensemble import DEFAULT_MEMBERS, DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

__all__ = ["DEFAULT_MEMBERS", "DeepEnsemble", "ProxyLandscape", "Surrogate"]
