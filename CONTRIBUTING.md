# Contributing

Thanks for your interest. This project is meant to be read as much as run, so
clarity of the code and of the reasoning behind it counts as much as
correctness.

## Setup

```bash
git clone https://github.com/SimonCrouzet/EvoFlowNet
cd EvoFlowNet
uv sync                     # GPU-first: installs the CUDA build of torch
uv run pre-commit install   # installs the pre-commit and commit-msg hooks
```

If you have no GPU, or you want the small environment:

```bash
uv sync --extra cpu         # 1.1GB instead of 4.7GB
```

A CUDA environment is around 4.7GB. To keep it off the disk holding the
checkout, point uv elsewhere:

```bash
export UV_PROJECT_ENVIRONMENT=/somewhere/with/room/evoflownet
```

## The loop

```bash
uv run pytest               # tests
uv run mypy                 # types, strict
uv run ruff check --fix .   # lint
uv run ruff format .        # format
uv run pre-commit run --all-files   # everything CI runs, locally
```

Anything CI enforces is also enforced by the pre-commit hooks, so a commit that
passes locally should not come back red.

## Pull requests

Work on a branch and open a pull request; `main` is protected and takes no
direct pushes. The `CI passed` check must be green before merge.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, `refactor:`,
`perf:`. The commit-msg hook enforces this, and the changelog is generated from
it, so the message is the changelog entry.

**Keep commits small and single-purpose.** One fix or one piece of functionality
per commit. A commit that renames things *and* changes behaviour cannot be
reviewed, reverted or bisected cleanly.

Write the commit body for someone who does not have the context you have right
now: say why, not what. The diff already says what.

## Standards for code

- **Public functions and classes need docstrings**, Google style. Ruff enforces
  their presence; it cannot enforce that they are useful.
- **Everything in `src/` is typed**, checked by mypy in strict mode.
- **Mathematical notation should match its source.** If you implement an
  equation from a paper, cite the paper and use its symbols. `Z`, `P_F`, `P_B`
  and `beta` are deliberately allowed as names — a reader checking the code
  against the paper is the point, and `partition_function` makes that harder.
  Lint rules N803 and N806 are disabled for this reason.

## Standards for tests

New behaviour needs a test that fails without the change.

Beyond that: where a correct answer is knowable, test against it rather than
against a plausible-looking output. This library is built on landscapes chosen
precisely because they make this possible.

- Ehrlich functions are constructed with a known optimum, so regret is exact.
- Small landscapes can be enumerated, so a sampler's distribution can be
  compared against `p*(x) ∝ R(x)^β` directly rather than eyeballed.
- Environment invariants — acyclicity, forward/backward action consistency,
  masked actions never being sampled — are property-tested with Hypothesis.
  These are where GFlowNet implementations actually break, and a test that only
  checks "it ran" will not catch any of them.

A sampler that runs, produces high-looking scores and silently samples the
wrong distribution passes a naive test suite. Please do not write that suite.

## Reporting things

Bugs and proposals go to
[Issues](https://github.com/SimonCrouzet/EvoFlowNet/issues). The templates ask
for a seed and a configuration because sampling bugs usually cannot be
reproduced without them.

Contributions are licensed under Apache-2.0, the same as the project.
