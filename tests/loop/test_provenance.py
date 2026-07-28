"""Tests for per-round campaign artifacts.

What these guard is that the chain is *navigable*, not merely written. A
directory of files nobody can order, or a batch file whose predictions were
silently dropped, is the same as having no provenance while looking like it has
some.
"""

import csv

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.core.types import Alphabet
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.loop import Campaign
from evoflownet.loop.provenance import FIELDS
from evoflownet.surrogate import DeepEnsemble
from evoflownet.tracking.base import Tracker

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 5


class Additive(FitnessLandscape):
    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    @property
    def optimum(self):
        return np.array([float(LENGTH)])

    def _evaluate(self, sequences):
        return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)


class Uniform(Sampler):
    def __init__(self, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def propose(self, n):
        self._count(n)
        return self._rng.integers(0, ALPHABET.size, size=(n, LENGTH), dtype=np.int32)


def campaign(tmp_path, *, rounds=3, surrogate=True):
    return Campaign(
        landscape=Additive(),
        sampler=Uniform(),
        surrogate=(
            DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=5, seed=0)
            if surrogate
            else None
        ),
        rounds=rounds,
        batch_size=6,
        pool_size=48,
        artifact_dir=tmp_path,
    )


class TestRoundFiles:
    def test_one_file_per_round(self, tmp_path):
        campaign(tmp_path).run()
        assert sorted(p.name for p in tmp_path.glob("round-*.csv")) == [
            "round-000.csv",
            "round-001.csv",
            "round-002.csv",
        ]

    def test_the_names_sort_into_run_order(self, tmp_path):
        # Zero-padding is what makes an artifact browser show round 2 before
        # round 10 -- without it the chain reads out of order, which is worse
        # than not recording it.
        campaign(tmp_path, rounds=3).run()
        names = sorted(p.name for p in tmp_path.glob("round-*.csv"))
        indices = [int(n.split("-")[1].split(".")[0]) for n in names]
        assert indices == sorted(indices)

    def test_a_batch_file_holds_every_design_measured(self, tmp_path):
        campaign(tmp_path, rounds=2).run()
        with (tmp_path / "round-000.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert tuple(rows[0]) == FIELDS

    def test_the_designs_are_recoverable(self, tmp_path):
        # The point of the file: six months on, someone wants the actual
        # sequence that was ordered, not a summary statistic of it.
        result = campaign(tmp_path, rounds=2).run()
        with (tmp_path / "round-000.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        recovered = np.array([[int(t) for t in r["sequence"].split()] for r in rows])
        assert np.array_equal(recovered, result.sequences[:6])

    def test_predictions_sit_beside_measurements(self, tmp_path):
        # Round 0 has no model, so its predictions are absent; later rounds
        # must carry them, because prediction disagreeing with measurement is
        # the most useful signal a round produces.
        campaign(tmp_path, rounds=3).run()
        with (tmp_path / "round-001.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert any(r["predicted"] not in {"nan", ""} for r in rows)

    def test_nothing_is_written_without_a_directory(self, tmp_path):
        Campaign(
            landscape=Additive(),
            sampler=Uniform(),
            rounds=2,
            batch_size=6,
            pool_size=48,
        ).run()
        assert not list(tmp_path.iterdir())


class TestManifest:
    def test_it_summarises_every_round(self, tmp_path):
        campaign(tmp_path, rounds=3).run()
        with (tmp_path / "rounds.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(r["index"]) for r in rows] == [0, 1, 2]

    def test_it_carries_the_ledger_columns(self, tmp_path):
        campaign(tmp_path, rounds=2).run()
        with (tmp_path / "rounds.csv").open() as handle:
            columns = set(next(csv.reader(handle)))
        assert {"feasible_fraction", "best_so_far", "surrogate_correlation"} <= columns

    def test_best_so_far_is_monotone_in_the_manifest(self, tmp_path):
        campaign(tmp_path, rounds=4).run()
        with (tmp_path / "rounds.csv").open() as handle:
            best = [float(r["best_so_far"]) for r in csv.DictReader(handle)]
        assert best == sorted(best)


class TestTrackerRegistration:
    def test_each_round_is_registered(self, tmp_path):
        class Recording(Tracker):
            def __init__(self):
                self.names = []

            def log_config(self, config):
                pass

            def log_metrics(self, metrics, *, step):
                pass

            def log_artifact(self, path, *, name):  # noqa: ARG002 - records the name only
                self.names.append(name)

        tracker = Recording()
        Campaign(
            landscape=Additive(),
            sampler=Uniform(),
            rounds=3,
            batch_size=6,
            pool_size=48,
            artifact_dir=tmp_path,
            tracker=tracker,
        ).run()
        assert tracker.names == ["round-000", "round-001", "round-002"]

    def test_a_local_run_still_leaves_a_trail(self, tmp_path):
        # tracker=noop is the debugging case, and it is exactly when the files
        # matter most, so they must not depend on a backend being configured.
        campaign(tmp_path, rounds=2, surrogate=False).run()
        assert (tmp_path / "round-000.csv").exists()
        assert (tmp_path / "rounds.csv").exists()


@pytest.mark.parametrize("rounds", [1, 5])
def test_it_works_at_any_number_of_rounds(tmp_path, rounds):
    campaign(tmp_path, rounds=rounds).run()
    assert len(list(tmp_path.glob("round-*.csv"))) == rounds
