# Getting started

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SimonCrouzet/EvoGFN
cd EvoGFN
uv sync                  # GPU: the CUDA build of torch, ~4.7 GB
uv sync --extra cpu      # CPU only, ~1.1 GB
```

Everything on the default path runs on CPU. A GPU helps for long sequences (`L = 256`) and
for the optional protein-language-model oracles (`--extra plm`).

A CUDA environment is around 4.7 GB. To keep it off the disk holding the checkout:

```bash
export UV_PROJECT_ENVIRONMENT=/somewhere/with/room/evogfn
```

!!! tip "Zero-install"
    The repository ships a [dev container](https://containers.dev/). Open it in VS Code or a
    Codespace and the environment builds itself with uv — no local Python needed. See
    `.devcontainer/devcontainer.json`.

---

## Run something in one minute

```bash
uv run evogfn campaign landscape=ehrlich
```

That runs a design–build–test–learn campaign: four rounds of 96 measured variants, a deep
ensemble surrogate refitted after each round, greedy acquisition, and a GFlowNet proposing.
It prints a per-round ledger.

`landscape=ehrlich` is closed-form and needs no network. The **default is `landscape=gb1`**,
which downloads ~3 MB of real deep-mutational-scanning data on first use and caches it.

Every part of the run is a Hydra override:

```bash
uv run evogfn campaign sampler=genetic acquisition=ucb selector=diverse
uv run evogfn campaign campaign.rounds=8 campaign.batch_size=48
uv run evogfn campaign tracker=noop
uv run evogfn campaign --help               # every configurable option
```

The groups you can swap are `landscape`, `env`, `reward`, `policy`, `training`, `objective`,
`surrogate`, `acquisition`, `selector`, `campaign` and `tracker`; `sampler` is a plain value
(`gflownet`, `genetic`, `hill-climb`, `random`).

To train a policy on its own, against the landscape rather than inside a campaign:

```bash
uv run evogfn train
uv run evogfn train landscape=gb1 training.steps=5000
uv run evogfn train reward.beta=1.0 tracker=noop
```

!!! warning "`train` and `campaign` charge the oracle very differently"
    `train` evaluates the landscape once per sampled trajectory — thousands of calls. That is
    fine when the landscape is closed-form and free, and catastrophic as a model of a wet-lab
    budget. `campaign` is the one that respects a budget: the sampler trains against a
    **surrogate proxy** and the oracle is charged only for the batch that is actually
    measured. If you are asking "how many assays would this have cost", `campaign` is the
    command that answers it.

---

## The same thing in Python

```python
from evogfn.acquisition.rules import Greedy, TopK
from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.loop.campaign import Campaign
from evogfn.models.policy import SequencePolicy
from evogfn.rewards.base import TemperedReward
from evogfn.surrogate.ensemble import DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

landscape = EhrlichLandscape(sequence_length=32, vocab_size=20, seed=7)

# The environment gets the landscape's feasibility rule. This is not optional
# decoration: omit `transitions` and nothing raises, every proposal scores -inf,
# and the surrogate has no finite value to fit.
env = MutationEnvironment(
    landscape.feasible_sequence(seed=0),
    landscape.alphabet,
    max_mutations=4,
    transitions=landscape.transition_matrix,
)

surrogate = DeepEnsemble(
    n_tokens=landscape.alphabet.size,
    sequence_length=landscape.sequence_length,
    seed=0,
)

sampler = GFlowNetSampler(
    env,
    SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        seed=0,
    ),
    # The proxy wraps the *same* surrogate instance the campaign refits, so the
    # sampler sees each round's model without the loop knowing it cares.
    proxy=ProxyLandscape(
        surrogate, alphabet=env.alphabet, sequence_length=env.sequence_length
    ),
    reward=TemperedReward(beta=3.0),
    config=TrainingConfig(steps=300, batch_size=64, seed=0),
    seed=0,
)

result = Campaign(
    landscape=landscape,
    sampler=sampler,
    surrogate=surrogate,
    acquisition=Greedy(),
    selector=TopK(),
    rounds=4,
    batch_size=96,
).run()

print(result.summary())
```

---

## Bring your own landscape

One interface. Implement `_evaluate` and two properties, and every sampler, campaign and
metric in the library works against it.

```python
import numpy as np

from evogfn.core.types import Alphabet
from evogfn.landscapes.base import FitnessLandscape


class MyAssay(FitnessLandscape):
    """Whatever you actually measure."""

    @property
    def alphabet(self) -> Alphabet:
        return Alphabet.protein()

    @property
    def sequence_length(self) -> int:
        return 42

    def _evaluate(self, sequences):
        # (n, sequence_length) token indices in, (n, n_objectives) values out.
        # The base class has already checked shape and token range.
        return my_model.predict(sequences)[:, None]
```

Override `is_feasible` as well if some sequences are not constructible, and hand the matching
constraint to your environment so it can mask rather than filter.

Three wrappers compose onto any landscape, and they are how a benchmark stays honest:

| Wrapper | What it enforces |
|---|---|
| `Budgeted` | a hard evaluation cap that raises rather than being quietly exceeded |
| `Noisy` / `SelectionNoisy` | measurement noise, so a method cannot rely on an exact oracle |
| `Cached` | one evaluation per distinct sequence, so re-measuring is not free |

---

## Development

```bash
uv run pytest                       # tests
uv run mypy                         # types, strict
uv run ruff check --fix .           # lint (covers notebooks/ too)
uv run ruff format .                # format
uv run pre-commit run --all-files   # everything CI runs, locally
```

Docs:

```bash
uv sync --group docs
uv run mkdocs serve                 # live-reloading preview on :8000
uv run mkdocs build                 # what CI checks; `strict: true`, so a broken link fails
```

Notebooks are `.py` "percent" scripts, so they run directly:

```bash
uv run python notebooks/01-landscapes-and-epistasis.py
uvx jupytext --to notebook notebooks/01-landscapes-and-epistasis.py   # if you want .ipynb
```

See [`CONTRIBUTING.md`](https://github.com/SimonCrouzet/EvoGFN/blob/main/CONTRIBUTING.md)
for conventions. The short version: public functions need Google-style docstrings that say
*why*, mathematical notation should match the paper it came from, and new behaviour needs a
test that fails without the change — preferably against a known correct answer, since the
landscapes here were chosen to make that possible.
