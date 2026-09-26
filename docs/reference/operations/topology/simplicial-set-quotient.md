# Finite simplicial-set congruence quotient

`topology.simplicial_set.quotient_by_congruence.compute` forms the degreewise
quotient of a finite truncated simplicial set. Each `degree_class_ids[n]` row
has one nonnegative class ID for every simplex in degree `n`; equal IDs define
the equivalence classes in that degree. IDs have no meaning across degrees.
Class IDs are JSON-safe nonnegative integers (at most `2**53 - 1`), so the
same labels transport losslessly as JSON numbers; their magnitude is
independent of the source simplex count.

The operation checks that every visible face and degeneracy sends equivalent
source simplices to equivalent target simplices. It then induces the face and
degeneracy maps on the quotient classes, checks all simplicial identities in
the output prefix, and returns the quotient together with its surjective
projection as a `TruncatedSimplicialMap`. The projection's fibers recover the
degreewise equivalence relation. The result makes no claim above the retained
maximum degree. The degree-set and operator presentation follows the standard
finite simplicial-set structure; see [Mathlib's simplicial-set
definition](https://leanprover-community.github.io/mathlib4_docs/Mathlib/AlgebraicTopology/SimplicialSet/Basic.html).

Class IDs are canonicalized by first occurrence in each source degree, so the
quotient labels are `q0`, `q1`, and so on. The source prefix is checked against
its face and degeneracy tables before the relation is used. Work for identity
replay, congruence scans, and induced table construction is bounded by the
source degree sizes and map rows (at most 11,000 admitted table steps); the
retained quotient and projection cardinality is preflighted against the
40,000-cell result bound. Incompatible class rows
return the first visible face or degeneracy that fails to descend.

For example, identifying the two vertices of `Delta[1]` and making the
corresponding identifications in its higher simplices yields a finite prefix
of the simplicial circle. Merging every simplex in each degree instead yields
the point simplicial set.
