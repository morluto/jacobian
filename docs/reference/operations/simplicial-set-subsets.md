# Finite simplicial subset prefixes

`topology.simplicial_set.subset.from_degree_families.compute` constructs a
subobject of one complete finite truncated simplicial set. The request gives
one strictly increasing list of source simplex indices in each degree. The
kernel checks that every selected simplex contains all of its visible faces
and degeneracies in the corresponding selected degree family. It then returns
the reindexed subobject prefix together with its degreewise injective inclusion
map into the exact ambient prefix.

Closure under all represented operators is the simplicial-set subobject
condition: in the presheaf description, a subobject is a subfunctor, so every
simplicial operator sends selected elements to selected elements. Mathlib
encodes this as `SSet.Subcomplex`; see the [Mathlib subcomplex reference](https://leanprover-community.github.io/mathlib4_docs/Mathlib/AlgebraicTopology/SimplicialSet/Subcomplex.html).

The value retains the ambient and selected prefixes through the typed inclusion
map and composes with existing simplicial-map operations after JSON
serialization. Its maximum degree remains the ambient prefix maximum; no claim
is made about higher degrees. Empty degree families are valid, including the
all-empty initial subobject. An invalid selected family is a typed domain
rejection, not an approximate subobject.
