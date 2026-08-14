# Finite posets

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Finite-poset operations use one typed finite-poset value throughout. The live
operations are:

- `poset.finite.compute` for canonical closure, Hasse reduction, extrema, and
  graded ranks;
- `poset.linear_extensions.count` for an exact bounded count;
- `poset.mobius_function.compute` for incidence-algebra Möbius values; and
- `poset.width.compute` for an exact maximum antichain and same-size chain
  partition.

Their values compose in ordinary requests and responses. There is no bulk
ledger, recurrence publication, or server-side replay state.
