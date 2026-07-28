"""Tests for the campaign command.

A console script that dispatches to a nonexistent module is a defect this
project has shipped once already: `twine check` and the wheel build both passed
it, and only a smoke test caught it. So the tests here run the command rather
than inspect it, and they cover the wiring most likely to rot -- the config
groups, the sampler names, and the feasibility matrix reaching the environment.
"""

import pytest

from evoflownet.cli.main import COMMANDS, SAMPLERS, main

BASE = [
    "campaign",
    "campaign.rounds=2",
    "campaign.batch_size=8",
    "campaign.pool_size=64",
    "training.steps=5",
    "training.batch_size=8",
    "surrogate.epochs=5",
    "tracker=noop",
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
]


def run(*overrides: str) -> int:
    return main([*BASE, *overrides])


class TestDispatch:
    def test_campaign_is_a_command(self):
        assert "campaign" in COMMANDS

    def test_the_usage_text_lists_it(self, capsys):
        main(["--help"])
        assert "campaign" in capsys.readouterr().out

    def test_an_unknown_command_is_refused(self, capsys):
        assert main(["nonsense"]) == 2
        assert "unknown command" in capsys.readouterr().err


class TestSamplers:
    @pytest.mark.parametrize("sampler", ["genetic", "hill-climb", "random"])
    def test_each_classical_sampler_runs(self, sampler, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run(f"sampler={sampler}", "landscape=ehrlich") == 0

    def test_the_gflownet_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run("sampler=gflownet", "landscape=ehrlich") == 0

    def test_every_advertised_sampler_is_buildable(self, tmp_path, monkeypatch):
        # A name in the help text that cannot be built is worse than an
        # undocumented one, because it reads as a promise.
        monkeypatch.chdir(tmp_path)
        for sampler in SAMPLERS:
            assert run(f"sampler={sampler}", "landscape=ehrlich") == 0

    def test_an_unknown_sampler_is_refused(self, tmp_path, monkeypatch, capsys):
        # Hydra catches the exception and exits, so the check is on the exit
        # rather than on the ValueError -- which is what a user actually sees.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            run("sampler=nonsense", "landscape=ehrlich")
        assert "unknown sampler" in capsys.readouterr().err


class TestConfigGroups:
    @pytest.mark.parametrize("acquisition", ["greedy", "ucb", "ei", "thompson"])
    def test_each_acquisition_rule_composes(self, acquisition, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run(f"acquisition={acquisition}", "sampler=random", "landscape=ehrlich") == 0

    @pytest.mark.parametrize("selector", ["topk", "diverse"])
    def test_each_selector_composes(self, selector, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run(f"selector={selector}", "sampler=random", "landscape=ehrlich") == 0

    @pytest.mark.parametrize("landscape", ["ehrlich", "gb1"])
    def test_each_landscape_composes(self, landscape, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run(f"landscape={landscape}", "sampler=random") == 0

    def test_the_protocol_is_overridable(self, tmp_path, monkeypatch, capsys):
        # The budget is what every claim is indexed by, so it has to be
        # settable from the command line rather than only in code.
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    *BASE[:1],
                    "campaign.rounds=3",
                    "campaign.batch_size=5",
                    "campaign.pool_size=64",
                    "sampler=random",
                    "landscape=ehrlich",
                    "surrogate.epochs=5",
                    "tracker=noop",
                    "hydra.run.dir=.",
                    "hydra.output_subdir=null",
                ]
            )
            == 0
        )
        assert "15 oracle calls" in capsys.readouterr().out


class TestFeasibility:
    def test_the_landscape_constraint_reaches_the_environment(self, tmp_path, monkeypatch, capsys):
        # The wiring whose absence silently switched masking off and made every
        # GFlowNet drown in -inf. A masked sampler on Ehrlich must report a
        # feasible fraction of exactly one.
        monkeypatch.chdir(tmp_path)
        assert run("sampler=gflownet", "landscape=ehrlich") == 0
        output = capsys.readouterr().out
        assert "feasible 1.000" in output
