# Preimage of a finite simplicial subobject

`topology.simplicial_set.map.preimage.compute` takes a complete finite-prefix
simplicial map `f : X -> Y` and a subobject `A -> Y`, and returns the
degreewise inverse image `f^-1(A) -> X` together with the restricted map
`f^-1(A) -> A`. In degree `n`, the selected simplices are exactly those
`x` in `X_n` for which `f_n(x)` belongs to `A_n`.

The result retains both input relations, the source inclusion, and the
restricted map, so the square can be composed and checked after serialization.
The defining relation is that composing the restricted map with `A -> Y`
equals composing `f^-1(A) -> X` with `f`. The construction follows the
preimage of a simplicial subcomplex in [Mathlib's simplicial-set subcomplex
API](https://leanprover-community.github.io/mathlib4_docs/Mathlib/AlgebraicTopology/SimplicialSet/Subcomplex.html).

The operation checks all visible simplicial identities and naturality squares
for both input maps. Naturality and subobject closure imply that the selected
families are closed under every represented face and degeneracy. Work and
result size are admitted before those families and their restricted tables
are built. The result is only for the common finite prefix and states nothing
above its maximum degree. A consumer that relies on the returned square after
JSON decoding must check the composite relation at its own admitted boundary.
