# Finite posets

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Finite-poset operations use one typed finite-poset value throughout. The live
operations are:

- `poset.finite.compute` for canonical closure, Hasse reduction, extrema, and
  graded ranks;
- `poset.maximal_chains.enumerate` for the complete inclusion-maximal chain
  family, with source endpoints, adjacent cover steps, and a length histogram;
- `poset.linear_extensions.count` for an exact bounded count;
- `poset.mobius_function.compute` for incidence-algebra Möbius values; and
- `poset.width.compute` for an exact maximum antichain and same-size chain
  partition.

Their typed results can be reused directly as inputs to later operations where
the contracts align.

Maximal-chain enumeration returns every inclusion-maximal chain exactly once,
ordered lexicographically by its source elements. A chain is maximal by
inclusion, so disconnected or non-graded posets can return chains of different
lengths. The empty poset has one empty chain; a nonempty poset never returns an
empty chain. Admission bounds the complete family by chain rows, repeated
element slots, dynamic-programming profile cells, and deterministic traversal
work before materializing any rows. These are mathematical allocation and work
limits, not transport-byte estimates.
