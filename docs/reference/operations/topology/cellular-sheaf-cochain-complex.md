# Cellular sheaf cochain complexes

The `cellular_sheaf.cochain_complex.compute` operation assembles the cochain
complex of one checked finite cellular sheaf on a finite simplicial complex and
exact coefficient field. It establishes one postcondition: the returned
coboundary matrices compose to zero, and they are the maps that the sheaf's own
cover restrictions induce.

The result is source-bound. It retains the input sheaf, the degree-indexed
cochain dimensions, the canonical cochain basis of each degree, and one
`CochainMatrix` per consecutive degree pair.

`coboundary_matrices[k]` maps the degree-`k` cochain basis to degree `k + 1`.
Rows use the target degree's cochain basis and columns use the source degree's,
so a cochain transported by this operation composes with the returned axes
without reindexing. Each cochain basis vector names one basis label of one
stalk, so the degree-`k` basis is the disjoint union of the stalk bases over the
degree-`k` simplices in canonical face order.

A simplicial complex of dimension `d` has `d + 1` cochain degrees, so
`cochain_dimensions` and `cochain_bases` each have one entry per degree from `0`
through `d`, and `coboundary_matrices` has one fewer entry than there are
degrees. A zero-dimensional complex has a single degree and no coboundary
matrix.

The zero product is checked, not assumed: the operation multiplies each
consecutive pair of coboundaries and refuses to construct the value unless every
product is the zero map. This is the mathematical content of the operation, not
a formatting check, and it is what makes the returned value usable as a
cochain complex by a downstream consumer.

This operation does not compute cohomology groups, does not reduce the
complex, and does not classify the sheaf. It returns the cochain-level
structure together with the exact field the matrices are written over. For the
cocycles, coboundaries, and cohomology of a sheaf, compose with
`cellular_sheaf.cohomology.compute`; for a natural morphism between two sheaves
and the induced degreewise cochain map, see
[cellular-sheaf-morphisms.md](cellular-sheaf-morphisms.md).

Sheaf admission and its bounds are shared with the other cellular sheaf
operations, so a sheaf accepted here is the same accepted carrier described in
[cellular-sheaf-morphisms.md](cellular-sheaf-morphisms.md).
