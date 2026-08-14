# Smith normal form and integral homology

[Documentation home](../../index.md)

This scenario demonstrates two bounded direct operations that retain the
mathematical data needed for inspection in their returned typed values:

- `matrix.normal_form.smith.certified.compute` returns a Smith diagonal and
  unimodular transformations (U,V) satisfying (D = UAV) for one integer
  matrix.
- `topology.simplicial_homology.integral.compute` returns the free ranks,
  torsion invariant factors, cycle generators, and bounding chains for one
  finite simplicial complex.

The diagonal entries of (D) are nonnegative and form the usual divisibility
chain. The transformations need not be canonical, but the stated matrix
relation and the invariant factors are directly checkable from the returned
value. Integral homology uses the same finite complex supplied in its request;
it exposes a finitely generated abelian-group decomposition, not persistent
homology or a homology ring.

Both operations are bounded. Read the live catalog for their current schemas
and limits, then pass the complete matrix or complex in the request. The server
does not create a complex workspace, store a proof record, or keep a replay
session after the response.
