"""Persisted results, so no campaign is ever run twice.

A suite run is hours of compute. Three things follow from that, and this module
exists for all three.

**It must survive interruption.** Each campaign is appended the moment it
finishes, so a run killed at hour six keeps everything up to hour six. Nothing
is held in memory waiting for a clean exit that may not come.

**It must resume at seed granularity.** Raising a tier from 30 seeds to 50 runs
twenty campaigns, not fifty. The key is ``(task, method, seed)``, which is the
finest unit that produces an independent number.

**It must know when a result is stale.** A cached campaign produced by code that
has since changed is worse than no cache: it silently mixes old numbers with new
ones inside a single table. Every record stores a fingerprint of the source that
produced it, and a record whose fingerprint no longer matches is re-run rather
than trusted.

What is fingerprinted, and what is not
--------------------------------------

The fingerprint is **per module**, one entry per ``.py`` file, and a record
stores only the modules its own run could have reached: the caller declares
entry points, and a static walk of the import graph expands them to their
transitive closure.

Per *package* was the obvious first answer and it was far too coarse. A record
was stale if any package it named had changed anywhere, so adding an unrelated
new file -- ``metrics/pareto.py``, say -- invalidated every genetic-algorithm
result in the store, none of which can reach it. Once that cost ~3,900 needless
campaigns, and it was avoided only by reasoning by hand about which code paths
were byte-identical and then calling :meth:`ResultStore.bless`. Hand-reasoning
about staleness is precisely what must not be load-bearing: it is unreviewable,
and it fails silently in the direction of trusting a stale number.

Nothing is excluded by name any more. The old scheme had to exempt
``benchmark`` itself -- adding a task or editing a report would otherwise have
invalidated every result in the store -- and the closure now does that job
properly: a report no campaign imports is simply never in one, while a task
definition that a campaign does reach is, which is right, since it decides what
was run.

The walk is static, over :mod:`ast`, and deliberately does not import anything:
importing has side effects, costs seconds per module here, and would need the
whole dependency tree installed just to decide whether a cache entry is valid.

What the walk cannot see
------------------------

A static walk is an over-approximation of what a run *reads* and an
under-approximation of what it *can reach*. The first is harmless -- the cost
is a re-run. The second is the failure mode that matters, so the honest limits
are these:

*Dynamic imports.* ``importlib.import_module(name)`` and any ``__import__``
with a computed name are invisible. So are Hydra's ``_target_`` strings, which
name a class in a YAML file and instantiate it by path -- a config-driven run
therefore depends on modules no import statement mentions.

*Dispatch by attribute.* ``getattr(sampler, "proxy_calls", 0)`` and registry
lookups keyed by string change behaviour without changing an import.

*Everything that is not Python.* Config YAML, downloaded datasets, and the
versions of torch and numpy underneath all change results and are not hashed.

*Parent ``__init__.py`` files, when nothing imports them by name.* Importing
``pkg.a.b`` really does execute ``pkg/a/__init__.py``, so following the letter
of the semantics would put every ancestor package in every closure. Measured
here, that takes a campaign's closure from 36 modules of 71 to 66 -- and puts
``metrics/pareto.py`` back in a genetic-algorithm record, which is the exact
failure this replaced. The judgement is that these files re-export and nothing
else; one that registered a handler or patched a default at import time would
break it, and that is the shape of edit to be careful with.

Against those, :meth:`ResultStore.stamp` is happy to over-include: an entry
point list that is too broad wastes compute, one that is too narrow yields a
table that silently mixes code versions. When in doubt, declare more.

Imports under ``if TYPE_CHECKING:`` are included for the same reason. They are
not read at runtime today, but promoting one to a runtime import is a one-line
change that would otherwise leave every record that depended on it looking
current.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: Packages the superseded per-package scheme hashed. Kept because records
#: written under it are still on disk and still readable: their ``source`` keys
#: are these bare names rather than dotted module names, and
#: :func:`package_fingerprint` is what they are compared against.
FINGERPRINTED = (
    "acquisition",
    "algorithms",
    "core",
    "env",
    "landscapes",
    "loop",
    "metrics",
    "models",
    "rewards",
    "surrogate",
)


def _package_root() -> Path:
    """Where the installed package lives."""
    return Path(__file__).resolve().parent.parent


def _modules(root: Path) -> dict[str, Path]:
    """Every module under a package directory, by dotted name.

    Args:
        root: The package directory. Its own name is the first component of
            every module name, so the mapping is what an import statement in
            this package would have to say.

    Returns:
        Source paths by dotted module name, a package's ``__init__.py`` being
        filed under the package's own name.
    """
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join((root.name, *parts))] = path
    return found


def fingerprint(root: Path | None = None) -> dict[str, str]:
    """Hash every module's source, one entry per file.

    Args:
        root: Package directory to hash. Defaults to the installed package.

    Returns:
        A short hash per dotted module name. The path that goes into the hash
        is relative to ``root``, and the files are sorted, so the same checkout
        digests identically on another machine and under another filesystem's
        iteration order -- otherwise a cache would invalidate on move alone.
    """
    base = _package_root() if root is None else root
    digests = {}
    for name, path in sorted(_modules(base).items()):
        digest = hashlib.sha256()
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(path.read_bytes())
        digests[name] = digest.hexdigest()[:16]
    return digests


def package_fingerprint(root: Path | None = None) -> dict[str, str]:
    """Hash each of the :data:`FINGERPRINTED` packages as a whole.

    This is the superseded scheme. It exists so that records written under it
    can still be judged stale or current, rather than crashing a load or --
    worse -- being read as current because their keys match nothing.

    Args:
        root: Package directory to hash. Defaults to the installed package.

    Returns:
        A short hash per bare package name, skipping any that is absent.
    """
    base = _package_root() if root is None else root
    digests = {}
    for package in FINGERPRINTED:
        directory = base / package
        if not directory.is_dir():
            continue
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(base).as_posix().encode())
            digest.update(path.read_bytes())
        digests[package] = digest.hexdigest()[:16]
    return digests


def _resolve(candidate: str, known: Mapping[str, Path]) -> str | None:
    """Longest prefix of a dotted name that is a module we hash.

    This is what settles ``from evogfn.x import y``, where ``y`` is either
    a submodule or a name defined inside ``x`` and the import statement alone
    cannot say which. If ``x/y.py`` exists the dependency is that submodule; if
    it does not, ``y`` came from ``x``'s own body and the dependency is ``x``.
    The same walk absorbs ``import evogfn.x.y`` and ``from x import *``.

    Args:
        candidate: A dotted name, not necessarily a module.
        known: Modules by dotted name.

    Returns:
        The module depended on, or ``None`` when the name is external -- which
        is how third-party and standard-library imports drop out.
    """
    while candidate:
        if candidate in known:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _base_of(node: ast.ImportFrom, package: str) -> str | None:
    """What a ``from ... import`` is relative to, resolved to an absolute name.

    Args:
        node: The import statement.
        package: Package containing the module that holds the statement.

    Returns:
        The dotted name the imported names hang off, or ``None`` when the
        statement escapes the package root and so cannot name one of ours.
    """
    if not node.level:
        return node.module
    base = package
    for _ in range(node.level - 1):
        base = base.rpartition(".")[0]
    if not base:
        return None
    return f"{base}.{node.module}" if node.module else base


def _imports_of(module: str, path: Path, known: Mapping[str, Path]) -> set[str]:
    """Modules a single source file imports.

    Walks the whole tree rather than the top level, which is what picks up
    imports inside functions, inside ``try`` blocks, and inside
    ``if TYPE_CHECKING:``.

    Args:
        module: Dotted name of the file being read.
        path: Its source path.
        known: Modules by dotted name.

    Returns:
        Dotted names, all of them in ``known``.

    Raises:
        SyntaxError: If the source does not parse. Left to propagate: a
            fingerprint computed by skipping a file it could not read would
            claim more than it knows.
    """
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _base_of(node, package)
            candidates = [] if base is None else [f"{base}.{alias.name}" for alias in node.names]
        else:
            continue
        found.update(target for name in candidates if (target := _resolve(name, known)) is not None)
    return found


def dependency_closure(entries: Iterable[str], root: Path | None = None) -> tuple[str, ...]:
    """Every module reachable from a set of entry points by import.

    Args:
        entries: Dotted module names the run starts from.
        root: Package directory to walk. Defaults to the installed package.

    Returns:
        The transitive closure, including the entry points themselves, sorted.

    Raises:
        ValueError: If an entry point is not a module. A typo would otherwise
            quietly shrink a record's dependency set, which is the one error
            this whole mechanism exists to make impossible.
    """
    base = _package_root() if root is None else root
    known = _modules(base)
    pending = list(entries)
    if unknown := sorted(name for name in pending if name not in known):
        raise ValueError(f"no such module under {base}: {', '.join(unknown)}")
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            # The visited set is also what terminates an import cycle, which
            # `__init__.py` re-exports make routine.
            continue
        seen.add(module)
        pending.extend(_imports_of(module, known[module], known) - seen)
    return tuple(sorted(seen))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One campaign's outcome, enough to rebuild any statistic from it.

    Attributes:
        task: Task name.
        method: Methodology name.
        seed: The seed this was run under.
        protocol: The protocol's repr, so a result cannot be read as though it
            came from a different budget.
        best: Best objective value measured.
        regret: Distance to the optimum, or ``None`` when unknown.
        diversity: Mean pairwise Hamming distance over everything measured.
        feasible_fraction: Share of the budget spent on constructible designs.
        oracle_calls: Calls actually charged.
        proposals: Candidates generated, including those never measured.
        trace: Best-so-far after each round.
        rounds: One dict per round, carrying what that round proposed,
            screened, measured and found. Kept because a surprising number is
            otherwise only diagnosable by re-running the campaign, which at
            L=256 costs minutes per seed and hours per arm. It is the
            round-to-round provenance the campaign already computes and would
            otherwise discard at this boundary.
        deterministic: Whether threading was pinned when this ran. A record
            made under multithreaded reduction cannot be reproduced, so it
            carries that fact rather than leaving a later reader to infer it.
        proxy_calls: Reward evaluations spent on the surrogate. Free against
            the oracle budget, not against wall clock, and reported so the
            trade is visible rather than implied.
        top_sequences: The best designs found, for inspection. Everything
            measured would be hundreds of megabytes across a suite; the best
            ten are what anyone actually looks at.
        source: Fingerprint of the code this run could reach, keyed by dotted
            module name. Records written before per-module hashing are keyed by
            bare package name instead; see :meth:`stale_against`.
    """

    task: str
    method: str
    seed: int
    protocol: str
    best: float
    regret: float | None
    diversity: float
    feasible_fraction: float
    oracle_calls: int
    proposals: int
    trace: list[float] = field(default_factory=list)
    rounds: list[dict[str, float]] = field(default_factory=list)
    proxy_calls: int = 0
    deterministic: bool = True
    top_sequences: list[list[int]] = field(default_factory=list)
    source: dict[str, str] = field(default_factory=dict)

    @property
    def per_package(self) -> bool:
        """Whether this was stamped by the superseded per-package scheme.

        Its keys are the bare package names in :data:`FINGERPRINTED`, so one of
        those settles it. "Dotless" would not: the root package is a module
        like any other and hashes under its own bare name.
        """
        return any(name in FINGERPRINTED for name in self.source)

    def stale_against(
        self,
        current: Mapping[str, str],
        legacy: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        """What this record depended on that has changed since it was produced.

        Only the modules the record stored are compared, so a change elsewhere
        in the tree leaves it alone -- that being the whole point of recording
        a dependency closure rather than a package list.

        Args:
            current: The live per-module fingerprint.
            legacy: The live per-package fingerprint, used only for records
                written under the superseded scheme. Computed on demand when
                omitted, which costs nothing for the records that do not need
                it and keeps an old record from being mistaken for a current
                one.

        Returns:
            The names that differ or have disappeared, empty when the record is
            current. Dotted module names, or bare package names for a record in
            the old format. A record with no stored fingerprint is treated as
            current, since it predates fingerprinting rather than contradicting
            it.
        """
        if not self.source:
            return ()
        if self.per_package:
            current = package_fingerprint() if legacy is None else legacy
        return tuple(
            sorted(name for name, digest in self.source.items() if current.get(name) != digest)
        )


class ResultStore:
    """Append-only storage, one file per task and method.

    Args:
        root: Directory to write under. Created if absent.

    Raises:
        ValueError: If the root exists and is not a directory.
    """

    def __init__(self, root: Path | str) -> None:
        """Prepare the storage directory."""
        self._root = Path(root)
        if self._root.exists() and not self._root.is_dir():
            raise ValueError(f"{self._root} exists and is not a directory")
        self._root.mkdir(parents=True, exist_ok=True)
        self._fingerprint = fingerprint()
        self._legacy: dict[str, str] | None = None
        self._closures: dict[tuple[str, ...], dict[str, str]] = {}

    @property
    def root(self) -> Path:
        """Where results are written."""
        return self._root

    def _legacy_fingerprint(self) -> dict[str, str]:
        """The per-package fingerprint, hashed once and only if it is wanted.

        A store holding nothing in the old format never pays for this; one that
        does pays for it once rather than once per record.
        """
        if self._legacy is None:
            self._legacy = package_fingerprint()
        return self._legacy

    def _changed(self, record: RunRecord) -> tuple[str, ...]:
        """What ``record`` depended on and no longer matches."""
        legacy = self._legacy_fingerprint() if record.per_package else None
        return record.stale_against(self._fingerprint, legacy)

    def _path(self, task: str, method: str) -> Path:
        """One file per task and method, so a rerun touches only what changed."""
        safe = method.replace("/", "-")
        return self._root / task / f"{safe}.jsonl"

    def load(self, task: str, method: str) -> dict[int, RunRecord]:
        """Every record held for one task and method, keyed by seed.

        A seed appearing more than once keeps the last entry, so re-running a
        seed overwrites rather than duplicating it.

        Args:
            task: Task name.
            method: Methodology name.

        Returns:
            Records by seed.
        """
        path = self._path(task, method)
        if not path.exists():
            return {}
        records: dict[int, RunRecord] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records[int(payload["seed"])] = RunRecord(**payload)
            except (json.JSONDecodeError, KeyError, TypeError):
                # A partial line is what an interrupted write leaves behind.
                # Skipping it costs one campaign; refusing to load would cost
                # the whole run.
                continue
        return records

    def usable(self, task: str, method: str) -> dict[int, RunRecord]:
        """Records that are current, dropping any the code has moved past.

        Args:
            task: Task name.
            method: Methodology name.

        Returns:
            Records by seed, excluding stale ones.
        """
        return {
            seed: record
            for seed, record in self.load(task, method).items()
            if not self._changed(record)
        }

    def missing(self, task: str, method: str, seeds: Sequence[int]) -> list[int]:
        """Which of ``seeds`` still need running.

        This is what makes raising a tier from 30 seeds to 50 cost twenty
        campaigns rather than fifty.

        Args:
            task: Task name.
            method: Methodology name.
            seeds: Seeds wanted.

        Returns:
            The subset not already held as a current result, in order.
        """
        held = self.usable(task, method)
        return [seed for seed in seeds if seed not in held]

    def stale(self, task: str, method: str) -> dict[int, tuple[str, ...]]:
        """Seeds whose stored result predates a change, and what changed.

        Args:
            task: Task name.
            method: Methodology name.

        Returns:
            Changed module names by seed, for the records that are stale.
        """
        return {
            seed: changed
            for seed, record in self.load(task, method).items()
            if (changed := self._changed(record))
        }

    def append(self, record: RunRecord) -> None:
        """Write one result immediately.

        Appending per campaign rather than per run is what makes an interrupted
        suite keep everything it had finished.

        Args:
            record: The result to store.
        """
        path = self._path(record.task, record.method)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")

    def dependencies(self, depends_on: Sequence[str]) -> dict[str, str]:
        """Fingerprint of the closure of some entry points.

        Args:
            depends_on: Dotted module names the run starts from -- typically
                the methodology's module, the landscape's module, and the loop
                that drives them. The caller is the only party that knows them.

        Returns:
            A hash per module in the transitive closure, entry points included.

        Raises:
            ValueError: If an entry point is not a module.
        """
        key = tuple(depends_on)
        if key not in self._closures:
            self._closures[key] = {
                name: self._fingerprint[name] for name in dependency_closure(key)
            }
        return dict(self._closures[key])

    def stamp(self, *, depends_on: Sequence[str] | None = None, **fields: object) -> RunRecord:
        """Build a record carrying the fingerprint of what produced it.

        Args:
            depends_on: Entry points of the run, expanded to their import
                closure and stored in place of the whole tree. Omitting it
                stores every module, which is correct but coarse: any edit
                anywhere then invalidates the record. Existing callers that
                pass nothing therefore keep working, they simply cache worse.
            **fields: Everything but ``source``.

        Returns:
            The record, ready to append.

        Raises:
            ValueError: If an entry point is not a module.
        """
        source = dict(self._fingerprint) if depends_on is None else self.dependencies(depends_on)
        return RunRecord(source=source, **fields)  # type: ignore[arg-type]

    def _restamp(self, record: RunRecord) -> RunRecord:
        """The same record, its stored dependencies refreshed to current.

        The key set is preserved rather than widened: blessing asserts that a
        change did not matter, not that the run depended on more than it did.
        """
        current = self._legacy_fingerprint() if record.per_package else self._fingerprint
        if not record.source:
            return replace(record, source=dict(self._fingerprint))
        refreshed = {name: current[name] for name in record.source if name in current}
        return replace(record, source=refreshed)

    def bless(self, task: str, method: str) -> int:
        """Restamp stored records as though the current code had produced them.

        For a change that provably cannot alter a completed run -- a new branch
        reached only where the old code raised, say.

        Per-module fingerprinting should make this rare. Adding a file, or
        editing one the run could not reach, no longer invalidates anything, so
        the cases left are genuine edits to a depended-on module that genuinely
        cannot change its output. Reaching for this often is a sign that the
        entry points passed to :meth:`stamp` are wider than the run.

        Use it only when that argument actually holds. It is the one operation
        here that can make a table mix results from different code, which is the
        failure the fingerprint exists to prevent.

        Args:
            task: Task name.
            method: Methodology name.

        Returns:
            How many records were restamped.
        """
        held = self.load(task, method)
        if not held:
            return 0
        path = self._path(task, method)
        lines = [json.dumps(asdict(self._restamp(record))) for _, record in sorted(held.items())]
        path.write_text("\n".join(lines) + "\n")
        return len(lines)

    def tasks(self) -> list[str]:
        """Task names with stored results."""
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    def methods(self, task: str) -> list[str]:
        """Methodology names stored for one task."""
        directory = self._root / task
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.jsonl"))

    def summarise(self, tasks: Iterable[str] | None = None) -> str:
        """What the store holds, and what of it is stale.

        Args:
            tasks: Task names to cover. Defaults to everything stored.

        Returns:
            A multi-line summary.
        """
        lines = [f"store: {self._root}"]
        for task in tasks if tasks is not None else self.tasks():
            for method in self.methods(task):
                held = self.load(task, method)
                stale = self.stale(task, method)
                note = (
                    f"  ({len(stale)} stale: {sorted({c for v in stale.values() for c in v})})"
                    if stale
                    else ""
                )
                lines.append(f"  {task}/{method}: {len(held)} seeds{note}")
        return "\n".join(lines)
