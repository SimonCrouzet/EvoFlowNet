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

The fingerprint covers the packages whose behaviour changes a number:
algorithms, landscapes, environments, models, rewards, surrogate, acquisition,
loop, metrics, core. It deliberately excludes ``benchmark`` itself -- adding a
task or editing a report would otherwise invalidate every result in the store,
which would make the cache useless exactly when it is most needed. What a task
*is* gets captured instead by its protocol and parameters, which are part of the
key.

The fingerprint is per package, so a report can say *which* component changed
rather than only that something did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Packages whose source affects a result. ``benchmark`` is excluded on purpose;
#: see the module docstring.
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


def fingerprint() -> dict[str, str]:
    """Hash each result-affecting package's source.

    Returns:
        A short hash per package name. Sorting the files makes the hash
        independent of directory iteration order, which otherwise varies
        between filesystems and would invalidate a cache on a different machine
        for no reason.
    """
    root = _package_root()
    digests = {}
    for package in FINGERPRINTED:
        directory = root / package
        if not directory.is_dir():
            continue
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        digests[package] = digest.hexdigest()[:16]
    return digests


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
        proxy_calls: Reward evaluations spent on the surrogate. Free against
            the oracle budget, not against wall clock, and reported so the
            trade is visible rather than implied.
        top_sequences: The best designs found, for inspection. Everything
            measured would be hundreds of megabytes across a suite; the best
            ten are what anyone actually looks at.
        source: Fingerprint of the code that produced this.
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
    top_sequences: list[list[int]] = field(default_factory=list)
    source: dict[str, str] = field(default_factory=dict)

    def stale_against(self, current: dict[str, str]) -> tuple[str, ...]:
        """Which fingerprinted packages have changed since this was produced.

        Args:
            current: The live fingerprint.

        Returns:
            Package names that differ, empty when the record is current. A
            record with no stored fingerprint is treated as current, since it
            predates fingerprinting rather than contradicting it.
        """
        if not self.source:
            return ()
        return tuple(
            sorted(
                name
                for name, digest in current.items()
                if name in self.source and self.source[name] != digest
            )
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

    @property
    def root(self) -> Path:
        """Where results are written."""
        return self._root

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
            if not record.stale_against(self._fingerprint)
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
            Changed package names by seed, for the records that are stale.
        """
        return {
            seed: changed
            for seed, record in self.load(task, method).items()
            if (changed := record.stale_against(self._fingerprint))
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

    def stamp(self, **fields: object) -> RunRecord:
        """Build a record carrying the current fingerprint.

        Args:
            **fields: Everything but ``source``.

        Returns:
            The record, ready to append.
        """
        return RunRecord(source=dict(self._fingerprint), **fields)  # type: ignore[arg-type]

    def bless(self, task: str, method: str) -> int:
        """Restamp stored records with the current fingerprint.

        For a change that provably cannot alter a completed run -- a new branch
        reached only where the old code raised, say. Blessing such a change
        keeps hours of valid compute that conservative invalidation would throw
        away.

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
        lines = [
            json.dumps(asdict(replace(record, source=dict(self._fingerprint))))
            for _, record in sorted(held.items())
        ]
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
