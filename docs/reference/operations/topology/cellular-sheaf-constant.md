# Constant cellular sheaves

[Documentation home](../../../index.md) · [Topology operations](index.md)

`cellular_sheaf.constant.compute` copies one exact based vector space to every
nonempty simplex in a finite simplicial complex. It returns the canonical
`FiniteCellularSheaf` value with identity restrictions on every face inclusion.
The common ordered basis labels identify the copied stalk bases; each map still
retains its source and target simplex axes.

The coefficient field is `QQ` or a bounded prime field `GF(p)`. The empty basis
is the zero vector space and produces empty `0 × 0` restriction matrices. A
single-vertex complex is a valid zero-dimensional base space. The current
`FiniteSimplicialComplex` carrier requires at least one vertex and facet, so an
empty complex is outside this operation's input representation.

Admission bounds the complex to 64 nonempty simplices, the stalk dimension to
8, the aggregate stalk rank to 512, cover restrictions to 512, derived
restrictions to 2,048, and all restriction-matrix cells to 65,536. It also
bounds the complete face-poset diamond checks, construction work, and estimated
exact serialized output. The restriction maps have coefficients only 0 and 1,
so no coefficient-growth phase is needed.

The result composes unchanged with `cellular_sheaf.sections.compute` and
`cellular_sheaf.cohomology.compute`. For a one-dimensional constant sheaf over
`GF(p)`, cellular sheaf cohomology agrees with ordinary simplicial cohomology;
tests compare both operation results on a point, interval, circle, and filled
triangle.
