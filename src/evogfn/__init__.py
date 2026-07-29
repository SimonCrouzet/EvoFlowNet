"""GFlowNets for in-silico directed evolution.

Directed evolution is usually framed as an optimization problem: find the single
best variant. This library treats it as a *sampling* problem instead — draw
variants in proportion to their fitness, so that a design round returns a diverse,
feasible set of candidates rather than many copies of one local optimum.

The public surface is organised around a small number of replaceable seams:

- `evogfn.landscapes` -- fitness functions (closed-form or empirical)
- `evogfn.env`        -- the state graph variants are built through
- `evogfn.algorithms` -- samplers: GFlowNets and classical baselines
- `evogfn.metrics`    -- evaluation, including exact-distribution checks
- `evogfn.loop`       -- design-build-test-learn campaigns under a budget
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evogfn")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
