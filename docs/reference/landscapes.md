# Landscapes

The functions being optimised against. A landscape is the only thing in the library a wet lab
would recognise as ground truth: sequences in, numbers out, and every call charged.

Two are built in, chosen because their correct answers are known — which means you can check
whether a method actually worked, not just whether it produced a plausible number.

::: evogfn.landscapes.base

::: evogfn.landscapes.ehrlich

::: evogfn.landscapes.gb1

::: evogfn.landscapes.trpb

## Wrappers

Measurement noise, an evaluation budget and caching compose onto any landscape, so a budget
cannot be accidentally bypassed and an exact oracle cannot be quietly assumed.

::: evogfn.landscapes.wrappers

## Types

::: evogfn.core.types
