# Minimal generators of a positive affine semigroup

[`affine_semigroup.minimal_generators.compute`](../tools.md) returns the unique
set of atoms (irreducible nonzero elements) of a finitely generated positive
affine semigroup. The result is itself a positive affine semigroup on the same
ambient row axis and grading. It also gives one exact factorization of every
source generator in the atom coordinates. Duplicate source columns therefore
map to one atom, while redundant generators retain a factorization in the
minimal set.

For a positive grading, each generator fiber is finite. A source generator is
reducible exactly when its complete fiber contains a factorization whose
coordinate sum is at least two. The operation checks those fibers, then factors
each source column in the atom configuration. This establishes both that the
returned atoms generate the source semigroup and that none is decomposable.

The operation admits each target's coefficient box using the existing exact
fiber bound, then admits the aggregate classification and transport work before
enumerating any fibers. It returns a complete exact atom set or a resource
refusal; it does not return a partial generator family.
