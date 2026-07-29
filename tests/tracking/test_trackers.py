"""Tests for run tracking and provenance."""

import io
import json

import pytest

from evogfn.tracking import (
    ConsoleTracker,
    MultiTracker,
    NoOpTracker,
    Tracker,
    git_provenance,
    run_provenance,
)


class Recording(Tracker):
    """A tracker that remembers what it was told, for asserting on."""

    def __init__(self, fail_on_finish=False):
        self.configs = []
        self.metrics = []
        self.artifacts = []
        self.finished = False
        self._fail_on_finish = fail_on_finish

    def log_config(self, config):
        self.configs.append(dict(config))

    def log_metrics(self, metrics, *, step):
        self.metrics.append((step, dict(metrics)))

    def log_artifact(self, path, *, name):
        self.artifacts.append((name, path))

    def finish(self):
        self.finished = True
        if self._fail_on_finish:
            raise RuntimeError("could not close")


class TestConsoleTracker:
    def test_metrics_are_emitted_on_the_interval(self):
        buffer = io.StringIO()
        tracker = ConsoleTracker(buffer, every=3)
        for step in range(7):
            tracker.log_metrics({"loss": float(step)}, step=step)
        emitted = [line for line in buffer.getvalue().splitlines() if line.startswith("step")]
        assert len(emitted) == 3  # steps 0, 3, 6

    def test_the_final_step_is_never_lost(self):
        # The last step of a run is usually the most interesting one, and a
        # plain interval is exactly what would skip it.
        buffer = io.StringIO()
        tracker = ConsoleTracker(buffer, every=10)
        for step in range(5):
            tracker.log_metrics({"loss": float(step)}, step=step)
        tracker.finish()
        last = buffer.getvalue().strip().splitlines()[-1].split()
        assert last[0] == "step"
        assert last[1] == "4"

    def test_it_does_not_repeat_an_already_emitted_final_step(self):
        buffer = io.StringIO()
        tracker = ConsoleTracker(buffer, every=2)
        tracker.log_metrics({"loss": 1.0}, step=4)
        tracker.finish()
        assert buffer.getvalue().count("step") == 1

    def test_finishing_without_any_metrics_is_harmless(self):
        buffer = io.StringIO()
        ConsoleTracker(buffer).finish()
        assert buffer.getvalue() == ""

    def test_the_configuration_is_printed_as_json(self):
        buffer = io.StringIO()
        ConsoleTracker(buffer).log_config({"beta": 3.0, "seed": 1})
        body = buffer.getvalue().split("config:\n", 1)[1]
        assert json.loads(body) == {"beta": 3.0, "seed": 1}

    def test_an_unserialisable_value_does_not_crash_the_run(self):
        # This is the very first call a run makes; failing here would lose the
        # whole run over a logging detail.
        buffer = io.StringIO()
        ConsoleTracker(buffer).log_config({"landscape": object()})
        assert "landscape" in buffer.getvalue()

    def test_it_writes_to_stderr_by_default(self, capsys):
        # So metrics do not contaminate anything a caller pipes from stdout.
        ConsoleTracker(every=1).log_metrics({"loss": 1.0}, step=0)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "loss" in captured.err

    def test_a_nonpositive_interval_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            ConsoleTracker(every=0)


class TestContextManager:
    def test_the_run_is_closed_on_a_clean_exit(self):
        recorder = Recording()
        with recorder:
            pass
        assert recorder.finished

    def test_the_run_is_closed_when_the_body_raises(self):
        # A crashed run is precisely when the partial record is wanted.
        recorder = Recording()
        with pytest.raises(RuntimeError, match="training blew up"), recorder:
            raise RuntimeError("training blew up")
        assert recorder.finished


class TestNoOpTracker:
    def test_it_accepts_everything_and_keeps_nothing(self, tmp_path):
        tracker = NoOpTracker()
        tracker.log_config({"a": 1})
        tracker.log_metrics({"loss": 1.0}, step=0)
        tracker.log_artifact(tmp_path / "x", name="x")
        tracker.finish()


class TestMultiTracker:
    def test_everything_is_forwarded_to_every_tracker(self, tmp_path):
        a, b = Recording(), Recording()
        tracker = MultiTracker(a, b)
        tracker.log_config({"beta": 3.0})
        tracker.log_metrics({"loss": 0.5}, step=7)
        tracker.log_artifact(tmp_path / "f", name="f")
        for recorder in (a, b):
            assert recorder.configs == [{"beta": 3.0}]
            assert recorder.metrics == [(7, {"loss": 0.5})]
            assert recorder.artifacts == [("f", tmp_path / "f")]

    def test_one_tracker_failing_to_close_does_not_prevent_the_others(self):
        # Losing one destination's record is a nuisance; losing all of them
        # because the first failed is a lost run.
        failing, healthy = Recording(fail_on_finish=True), Recording()
        with pytest.raises(ExceptionGroup):
            MultiTracker(failing, healthy).finish()
        assert healthy.finished

    def test_no_trackers_is_allowed(self):
        MultiTracker().log_metrics({"loss": 1.0}, step=0)


class TestProvenance:
    def test_it_records_what_a_result_needs_to_be_reproducible(self):
        provenance = run_provenance(seed=3)
        assert provenance["seed"] == 3
        assert provenance["evogfn_version"]
        assert provenance["python"]
        assert "git" in provenance

    def test_the_dirty_flag_is_recorded_explicitly(self):
        # A commit hash alone claims "this code produced this number". With
        # uncommitted changes that claim is false, and nothing else would say so.
        git = git_provenance()
        assert set(git) == {"commit", "branch", "dirty"}
        if git["commit"] is not None:
            assert isinstance(git["dirty"], bool)

    def test_it_reports_rather_than_raising_outside_a_repository(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        git = git_provenance()
        # Either genuinely outside a repo, or inside one that contains tmp_path.
        assert git["commit"] is None or isinstance(git["commit"], str)
