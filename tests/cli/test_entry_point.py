"""Tests for the installed command line entry point.

The console script declared in pyproject.toml previously pointed at a module
that did not exist, so ``pip install evogfn && evogfn`` failed with a
ModuleNotFoundError. Neither the build job nor ``twine check`` caught it,
because neither runs the script. These tests do.
"""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

import evogfn.cli.main as cli_module
from evogfn.cli.main import _USAGE, COMMANDS, main


def run_cli(*arguments, cwd=None):
    """Invoke the entry point the way an installed user would."""
    return subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [sys.executable, "-m", "evogfn.cli.main", *arguments],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=cwd,
        check=False,
    )


class TestTheScriptActuallyRuns:
    def test_the_declared_entry_point_is_importable(self):
        # The exact failure that shipped: the console script named a module that
        # did not exist. Importing it is the cheapest possible guard.
        assert callable(cli_module.main)

    def test_invoking_it_with_no_arguments_prints_usage(self, capsys):
        assert main([]) == 0
        assert "usage: evogfn" in capsys.readouterr().out

    def test_help_is_not_an_error(self, capsys):
        assert main(["--help"]) == 0
        assert "commands:" in capsys.readouterr().out

    def test_an_unknown_command_is_rejected_with_a_nonzero_code(self, capsys):
        assert main(["frobnicate"]) == 2
        assert "unknown command" in capsys.readouterr().err

    def test_the_usage_text_and_the_dispatcher_agree(self):
        # Usage listing a command the dispatcher rejects would be worse than no
        # usage at all.
        assert COMMANDS
        for command in COMMANDS:
            assert command in _USAGE

    def test_the_module_runs_as_a_subprocess(self):
        # Closest thing to what an installed user does, without needing an
        # install: catches import-time errors the in-process test cannot.
        result = run_cli()
        assert result.returncode == 0
        assert "usage: evogfn" in result.stdout


@pytest.mark.slow
class TestTraining:
    def test_a_short_run_completes_and_reports(self, tmp_path):
        result = run_cli(
            "train",
            "training.steps=20",
            "training.batch_size=8",
            "training.log_every=10",
            "landscape.sequence_length=8",
            "landscape.vocab_size=4",
            "landscape.motif_length=2",
            "env.max_mutations=2",
            f"hydra.run.dir={tmp_path / 'run'}",
        )
        assert result.returncode == 0, result.stderr
        # Metrics go to stderr by design, so stdout stays clean for piping.
        assert "step" in result.stderr
        assert "oracle_calls" in result.stderr

    def test_the_run_records_its_own_provenance(self, tmp_path):
        # A metric with no record of the code that produced it is not a result.
        result = run_cli(
            "train",
            "training.steps=5",
            "training.batch_size=4",
            "landscape.sequence_length=8",
            "landscape.vocab_size=4",
            "landscape.motif_length=2",
            "env.max_mutations=2",
            f"hydra.run.dir={tmp_path / 'run'}",
        )
        assert result.returncode == 0, result.stderr
        for field in ("evogfn_version", "git", "commit", "dirty", "seed"):
            assert field in result.stderr

    def test_a_component_can_be_swapped_from_the_command_line(self, tmp_path):
        # The point of the config groups: selecting a different tracker must not
        # require touching code.
        result = run_cli(
            "train",
            "tracker=noop",
            "training.steps=5",
            "training.batch_size=4",
            "landscape.sequence_length=8",
            "landscape.vocab_size=4",
            "landscape.motif_length=2",
            "env.max_mutations=2",
            f"hydra.run.dir={tmp_path / 'run'}",
        )
        assert result.returncode == 0, result.stderr
        assert "step" not in result.stderr


class TestPackagedConfiguration:
    def test_the_configs_ship_with_the_package(self):
        # Hydra composes a run from these at import time of the installed
        # package. If they are not packaged, the CLI works from a checkout and
        # fails for everyone who installs it -- the worst kind of defect.
        configs = files("evogfn.configs")
        assert configs.joinpath("train.yaml").is_file()
        for group in ("landscape", "env", "reward", "policy", "training", "tracker"):
            assert configs.joinpath(group).is_dir(), f"missing config group {group}"

    def test_every_config_group_referenced_by_the_default_run_exists(self):
        configs = Path(str(files("evogfn.configs")))
        text = (configs / "train.yaml").read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and ":" in stripped and "_self_" not in stripped:
                group, _, choice = stripped[2:].partition(":")
                assert (configs / group.strip() / f"{choice.strip()}.yaml").is_file()
