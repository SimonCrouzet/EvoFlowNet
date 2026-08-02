"""Making a benchmark reproduce its own numbers.

A benchmark that cannot reproduce itself invalidates everything downstream of
it. Multithreaded torch cannot: two runs of an *identical* configuration in the
*same process* can land on different designs. At one thread they are
bit-identical.

The cause is not a seeding bug. It is floating-point reduction order: a
multithreaded matmul sums its partial products in whatever order the threads
finish, and addition is not associative in floating point. The difference is
about 1e-16 per operation, which a few hundred gradient steps amplify into a
different trajectory and eventually a different design.

Why this is easy to miss
------------------------

The classical baselines are numpy and bit-identical across replicates, so a
spot-check of the harness looks clean. Only the torch arms drift, and they
drift *within* the range of genuine seed-to-seed variation -- so the numbers
stay plausible and the comparison quietly stops being paired.

Two pools, not one
------------------

``torch.set_num_threads`` binds the pool torch dispatches its own kernels on.
It says nothing about the BLAS that numpy links, which in a wheel install is a
*separate* copy of OpenBLAS with its own thread count. numpy reductions and
``@`` therefore stay multithreaded while torch is pinned, and they carry the
same non-associativity, so a result can be nondeterministic through the numpy
path with the torch path fully pinned.

The obvious lever -- ``OMP_NUM_THREADS`` and friends -- does not reach that
pool from inside Python. Those variables are read once, when the library is
dlopen'd, which for numpy's BLAS is at ``import numpy``: long before any
function in this module runs. Setting them from Python is not a weaker fix, it
is no fix at all for anything already imported, and because
``torch.get_num_threads`` keeps returning one, the failure reports success.

Hence two mechanisms, covering disjoint cases, plus a check that trusts
neither:

* ``threadpool_limits`` reaches into the pools that are *already loaded* and
  resizes them in place. This is the only thing that touches numpy's BLAS.
* The environment variables still cover libraries that are *not yet* loaded --
  a later import in this process, or a child process -- because those do read
  the variables at their own load time.
* ``is_deterministic`` re-inspects the live pools rather than a flag, so a pool
  that neither mechanism reached reads as not-verified instead of fine.

The cost
--------

One thread per process. That is not a real loss here: the suite parallelises
across tasks, so throughput comes from running many processes rather than from
threading one, and most of the work is serial Python rather than matmul.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import torch
from threadpoolctl import threadpool_info, threadpool_limits

#: Environment variables that set thread counts in the numerical libraries
#: underneath torch and numpy.
#:
#: These do **not** affect pools that are already loaded, which in practice
#: means they do not affect numpy's BLAS: it is dlopen'd at ``import numpy``,
#: reads its count once at that moment, and ignores every later write. Nobody
#: should re-derive that the hard way -- ``threadpool_limits`` in
#: ``configure_determinism`` is what resizes loaded pools.
#:
#: They are still set, for the two cases where a library reads them *after* we
#: write them: an import that happens later in this process, and any child
#: process, which inherits the environment and loads its own copies fresh.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: The limiter returned by the last ``configure_determinism`` call, held only
#: to make the lifetime deliberate. ``threadpool_limits`` restores the original
#: counts when it is used as a context manager and never otherwise, so the
#: limits outlive this reference either way; keeping it means the decision to
#: apply them process-wide is visible in the code rather than implied by an
#: ignored return value.
_LIMITER: Any = None


def configure_determinism(*, threads: int = 1) -> None:
    """Pin every thread pool this process can reach so a seeded run repeats.

    Call before any tensor work. Idempotent.

    The limit is applied globally rather than as a context manager around a
    campaign. Both are available -- ``threadpool_limits`` is a context manager
    -- and global is the right one here because of where the callers sit: the
    experiment entry points call this once at the top of ``main`` and then run
    the whole process's work under it, and the record-writing sites call
    ``is_deterministic`` deep inside the campaign, far from any scope this
    could open. A context manager would have to wrap the body of every entry
    point to cover the same ground, and would silently cover less than it
    looked like it did the first time work moved outside the block.

    The trade the global form makes is that a library imported *after* this
    runs brings its own unpinned pool, which no already-applied limit can
    catch. That is why the environment variables are still set -- they are read
    at load time, so they do catch the late import -- and, more importantly,
    why ``is_deterministic`` inspects live pools instead of trusting that this
    function ran.

    Args:
        threads: Threads to allow. Anything above one reintroduces the
            reduction-order nondeterminism this exists to remove; it is a
            parameter only so a caller who has decided speed matters more than
            reproducibility has to say so explicitly.
    """
    # Not effective for this process's already-loaded libraries; see
    # THREAD_VARIABLES. Kept for later imports and for child processes.
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    torch.set_num_threads(threads)
    # Interop threads govern how independent ops are dispatched. Torch refuses
    # to change this once parallel work has started, which is not an error
    # worth failing a run over -- the intra-op setting above is what carries
    # the reduction order.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(threads)
    # The one call that reaches numpy's BLAS. Resizes the pools that are loaded
    # right now, which is why it comes last: anything the imports above pulled
    # in is included.
    global _LIMITER  # noqa: PLW0603 -- process-wide state, deliberately
    _LIMITER = threadpool_limits(limits=threads)


def thread_pools() -> tuple[tuple[str, int | None], ...]:
    """The thread count of every numerical pool loaded in this process.

    Reported per pool rather than summarised because the pools fail
    independently: torch's OpenMP and numpy's BLAS are separate libraries with
    separate counts, and the failure this module exists to prevent is exactly
    one of them being pinned while the other is not.

    Returns:
        One ``(name, threads)`` pair per pool, in the order threadpoolctl
        found them. ``threads`` is ``None`` when the pool was found but its
        count could not be read, which is a distinct outcome from a pool that
        reports a count: it means unverified, not unlimited and not fine. An
        empty tuple means introspection found nothing at all, which for a
        process that has imported torch is itself a failure to verify rather
        than evidence of no threading.
    """
    pools: list[tuple[str, int | None]] = []
    for pool in threadpool_info():
        name = pool.get("prefix") or pool.get("internal_api") or "unknown"
        count = pool.get("num_threads")
        pools.append((str(name), count if isinstance(count, int) else None))
    return tuple(pools)


def is_deterministic() -> bool:
    """Whether the current process is configured to reproduce its runs.

    Interrogates the live pools rather than the torch counter alone. The
    distinction matters because the two disagree in the case that actually
    occurred: ``torch.get_num_threads`` returns one whenever
    ``configure_determinism`` ran, including when it failed to reach numpy's
    BLAS, so a check built on it certifies a condition it never tested. Every
    stored record carries this value as its determinism flag, which makes an
    over-broad reading of it a claim in the results rather than a local
    inaccuracy.

    Anything unverifiable counts as not deterministic: a pool whose count
    cannot be read, and the case where no pool can be found at all. The
    asymmetry is deliberate -- a run wrongly marked nondeterministic costs a
    rerun, a run wrongly marked deterministic costs whatever was concluded
    from it.

    Returns:
        ``True`` when torch's intra-op threading is pinned to one *and* every
        pool found reports exactly one thread. Reported rather than assumed so
        a result can record the conditions it was produced under.
    """
    if torch.get_num_threads() != 1:
        return False
    pools = thread_pools()
    if not pools:
        return False
    return all(count == 1 for _, count in pools)
