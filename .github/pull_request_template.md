<!--
Keep this short. CI already checks formatting, types and tests, so this
template only asks for what CI cannot: why the change is right.
-->

## What and why

<!-- What changes, and what problem it solves. Link the issue if there is one. -->

## How this was verified

<!--
Not "tests pass" -- CI reports that. What did you actually check?
For anything touching sampling, training or a landscape, state which
correctness property you confirmed and how. For example:

- exact-distribution check on an enumerable landscape (L1 vs p*(x) proportional to R(x)^beta)
- forward/backward action consistency on the mutation environment
- regret against the known optimum of an Ehrlich instance
- benchmark numbers before and after, at equal evaluation budget
-->

## Numerical impact

<!--
Does this change results anyone has already reported or plotted?
If yes, say which and by how much. If no, say "none" -- that is a real answer
and reviewers need it stated rather than assumed.
-->

## Checklist

- [ ] Commits follow Conventional Commits and are scoped to one change each
- [ ] New behaviour has a test that fails without the change
- [ ] Public functions and classes have docstrings
- [ ] Anything taken from a paper cites it, and matches its notation
