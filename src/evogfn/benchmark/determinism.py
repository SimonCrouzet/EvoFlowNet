"""Making a benchmark reproduce its own numbers.

A benchmark that cannot reproduce itself invalidates everything downstream of
it, and this one could not. Measured: with ``torch.set_num_threads(4)``, two
runs of an *identical* configuration in the *same process* returned 0.125 and
0.375 -- one full reward-quantisation level apart. At one thread they are
bit-identical.

The cause is not a seeding bug. It is floating-point reduction order: a
multithreaded matmul sums its partial products in whatever order the threads
finish, and addition is not associative in floating point. The difference is
about 1e-16 per operation, which a few hundred gradient steps amplify into a
different trajectory and eventually a different design.

Why this was invisible
----------------------

The classical baselines are numpy and bit-identical across replicates, so a
spot-check of the harness looks clean. Only the torch arms drift, and they
drift *within* the range of genuine seed-to-seed variation -- so the numbers
stay plausible and the comparison quietly stops being paired. The symptom that
exposed it was three accidental replicates of one configuration disagreeing by
about as much as the effect being measured.

The cost
--------

One thread per process. That is not a real loss here: the suite parallelises
across tasks, so throughput comes from running twelve processes rather than
from threading one, and the earlier measurement of core utilisation said the
same thing -- most of the work is serial Python, not matmul.
"""

from __future__ import annotations

import contextlib
import os

import torch

#: Environment variables that set thread counts in the numerical libraries
#: underneath torch and numpy. Setting them is necessary but not sufficient:
#: they are read at import, so a process that imported torch before they were
#: set keeps whatever it started with. ``torch.set_num_threads`` is what
#: actually binds, which is why both are done.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_determinism(*, threads: int = 1) -> None:
    """Pin thread counts so a seeded run reproduces exactly.

    Call before any tensor work. Idempotent.

    Args:
        threads: Threads to allow. Anything above one reintroduces the
            reduction-order nondeterminism this exists to remove; it is a
            parameter only so a caller who has decided speed matters more than
            reproducibility has to say so explicitly.
    """
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    torch.set_num_threads(threads)
    # Interop threads govern how independent ops are dispatched. Torch refuses
    # to change this once parallel work has started, which is not an error
    # worth failing a run over -- the intra-op setting above is what carries
    # the reduction order.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(threads)


def is_deterministic() -> bool:
    """Whether the current process is configured to reproduce its runs.

    Returns:
        ``True`` when intra-op threading is pinned to one. Reported rather than
        assumed so a result can record the conditions it was produced under.
    """
    return torch.get_num_threads() == 1
