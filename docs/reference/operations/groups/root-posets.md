# Positive-root posets

`root_system.root_poset.compute` returns the complete poset on a finite
crystallographic root system's positive roots. The result includes a canonical
`FinitePoset`, its positive-root coordinate axis, and the finite Cartan datum
whose ordered simple-root basis interprets those coordinates. Poset labels
`root_00`, `root_01`, and so on index roots in lexicographic coordinate order.

For positive roots `alpha` and `beta`, `alpha < beta` exactly when `beta -
alpha` has nonnegative integer coordinates in the simple-root basis and
`alpha != beta`. The output includes the full strict order, its cover relation,
incomparable pairs, and canonical ranks when the poset is graded.

The operation admits at most 64 positive roots, matching the existing finite
poset carrier. It rejects larger root families before pairwise order
comparisons and poset materialization. At the boundary, there are at most
`64*63/2` unordered root pairs, at most `64*64*8` coordinate comparisons, at
most `64^3` cover-reduction checks, and at most 64 elements / 4,096 relation
slots in the emitted poset. Thus B8's 64 positive roots are accepted; E8's 120
are rejected for this operation even though other root-system operations
support E8.

The root-index axis and Cartan datum are retained alongside the generic poset,
so existing poset consumers can act on `result.poset` while root coordinates
remain recoverable without reconstructing a basis convention.

[Operation references](../index.md) · [Tool surface](../../tools.md)
