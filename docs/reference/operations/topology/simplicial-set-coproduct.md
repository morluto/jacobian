# Finite simplicial-set coproducts

`topology.simplicial_set.coproduct.compute` constructs the degreewise disjoint
union of two finite truncated simplicial-set prefixes. Both inputs must cover
the same maximum degree `N`; in every retained degree it forms
`(X ⊔ Y)_n = X_n ⊔ Y_n`.

The output labels are factor-tagged (`L0`, `L1`, ..., then `R0`, `R1`, ...)
within each degree. The `tagged_axes` field preserves the exact factor identity
and input simplex index for each output simplex. The result also retains both
factor values and returns `left_inclusion` and `right_inclusion` as typed
`TruncatedSimplicialMap` values, ready for existing map checks and composition.

Each face and degeneracy acts on the component containing its simplex and keeps
the same factor tag. Target indices for right-factor simplices are offset by the
left factor's size in the target degree. The simplicial identities therefore
hold componentwise, and the inclusions are natural transformations.

Admission occurs before constructing tagged axes or map tables. The operation
bounds each degree by 32 simplices, the complete prefix by 96 simplices, map
rows by 50,000, and conservative serialized output size by 1,000,000 bytes.
Requests above those limits fail as resource-admission errors without
returning a partial coproduct. The finite carrier currently requires every
degree to be nonempty, so inputs with empty degree levels are outside its
represented domain.
