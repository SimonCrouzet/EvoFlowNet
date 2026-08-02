"""Tests for the hyperparameter sensitivity arms.

A sweep is worth its compute only if the axis it names is the axis it moves.
The failure this guards against is not a crash but a flat column: nine arms that
all build the same campaign produce nine indistinguishable rows, and a reader
concludes the setting does not matter when what happened is that it was never
varied. So the tests here are about *difference* -- that each value reaches the
object it is supposed to configure, and that the arm names stay legible enough
for a report to group them by axis.

They are deliberately structural rather than behavioural. Whether ``beta=10``
searches better than ``beta=1`` is what the benchmark run answers, and asserting
it here would pin a result rather than the machinery that measures it.
"""

import pytest

from evogfn.benchmark.methods import (
    BASELINES,
    DEFAULT_BETA,
    DEFAULT_MIX,
    DEFAULT_TRAINING_STEPS,
    OBJECTIVES,
    SENSITIVITY_GRID,
    sensitivity,
)
from evogfn.benchmark.suite import objective_task


@pytest.fixture(scope="module")
def arms():
    return sensitivity()


@pytest.fixture(scope="module")
def task():
    return objective_task()


class TestGrid:
    def test_every_axis_brackets_its_default(self):
        # A one-sided grid cannot distinguish "the default is good" from "the
        # grid stopped before it got worse", and the two read identically in a
        # table. Bracketing is what makes a flat column interpretable.
        defaults = {
            "steps": float(DEFAULT_TRAINING_STEPS),
            "beta": DEFAULT_BETA,
            "mix": DEFAULT_MIX,
        }
        for axis, values in SENSITIVITY_GRID.items():
            assert min(values) < defaults[axis] < max(values), axis

    def test_the_shipped_default_is_itself_a_point_on_every_axis(self):
        # Without this the sweep reports on a neighbourhood of the shipped
        # configuration without ever measuring it, so nothing in the table is
        # the number the headline rows were produced at.
        defaults = {
            "steps": float(DEFAULT_TRAINING_STEPS),
            "beta": DEFAULT_BETA,
            "mix": DEFAULT_MIX,
        }
        for axis, values in SENSITIVITY_GRID.items():
            assert defaults[axis] in values, axis

    def test_every_axis_names_a_knob_the_methodology_accepts(self, arms):
        # `mix` lived on GeneticConfig for a while without being reachable from
        # methods.py, so a grid could name it and silently sweep nothing.
        assert len(arms) == sum(len(v) for v in SENSITIVITY_GRID.values())


class TestNames:
    def test_each_name_carries_its_axis_and_value(self, arms):
        for axis, values in SENSITIVITY_GRID.items():
            for value in values:
                assert f"{axis}-{value:g}" in arms

    def test_the_names_are_stable_across_calls(self, arms):
        # These are store keys. A name that moved would orphan every result
        # already computed under the old one and quietly re-run the tier.
        assert list(sensitivity()) == list(arms)

    def test_no_name_collides_with_a_headline_arm(self, arms):
        assert not set(arms) & (set(BASELINES) | set(OBJECTIVES))


class TestTheAxesActuallyMove:
    """Each value must reach the object it configures, not just the factory.

    These reach past the sampler's public surface on purpose. A methodology's
    settings are not observable from outside -- every arm here reports the same
    ``name`` -- so the only way to show that an arm was configured rather than
    merely constructed is to look at what it was configured with.
    """

    def test_steps_reaches_the_training_config(self, arms, task):
        built = {
            value: arms[f"steps-{value:g}"](task, 0).sampler._config.steps
            for value in SENSITIVITY_GRID["steps"]
        }
        assert built == {value: int(value) for value in SENSITIVITY_GRID["steps"]}

    def test_beta_reaches_the_reward(self, arms, task):
        built = {
            value: arms[f"beta-{value:g}"](task, 0).sampler._reward.beta
            for value in SENSITIVITY_GRID["beta"]
        }
        assert built == {value: value for value in SENSITIVITY_GRID["beta"]}

    def test_mix_reaches_the_genetic_config(self, arms, task):
        built = {
            value: arms[f"mix-{value:g}"](task, 0).sampler._genetic_config.mix
            for value in SENSITIVITY_GRID["mix"]
        }
        assert built == {value: value for value in SENSITIVITY_GRID["mix"]}

    def test_only_the_swept_axis_moves(self, arms, task):
        # The whole design is one-at-a-time, so an arm that also perturbed a
        # second setting would attribute that setting's effect to this one.
        for value in SENSITIVITY_GRID["beta"]:
            sampler = arms[f"beta-{value:g}"](task, 0).sampler
            assert sampler._config.steps == DEFAULT_TRAINING_STEPS

    def test_every_arm_builds_a_runnable_campaign(self, arms, task):
        for name, methodology in arms.items():
            campaign = methodology(task, 0)
            assert campaign.budget == task.protocol.rounds * task.protocol.batch_size, name
