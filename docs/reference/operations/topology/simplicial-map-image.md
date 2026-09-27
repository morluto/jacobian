# Image factorization of a finite simplicial map

`topology.simplicial_set.map.image.compute` takes a complete finite-prefix
`TruncatedSimplicialMap` and returns its image in each degree, the induced
surjection onto that image, and the inclusion of the image into the target.
The image retains the target's simplex labels in target order. The operation
checks the visible simplicial identities of both carriers and every face and
degeneracy square of the supplied map before constructing the image.

The family of degreewise images is a simplicial subset: if `y = f_n(x)`, then
`d_i(y) = f_(n-1)(d_i(x))` and `s_i(y) = f_(n+1)(s_i(x))`, so every visible
face and degeneracy of an image simplex remains in the image. Restricting the
target tables therefore defines the image simplicial set, and the factor maps
compose degreewise to the original map. This is the finite-prefix form of the
image subcomplex and factorization used by [Mathlib's simplicial-set
subcomplex API](https://leanprover-community.github.io/mathlib4_docs/Mathlib/AlgebraicTopology/SimplicialSet/Subcomplex.html).

All identity replay, naturality checks, image construction, and output
serialization are admitted against fixed finite-prefix bounds before the
image tables are built. The result makes no claim above the common maximum
degree of the input map. A JSON round trip preserves the typed carriers and
factor maps. As with other serialized mathematical values, a consumer should
recheck the composite relation if a later operation relies on that authored
factorization.
