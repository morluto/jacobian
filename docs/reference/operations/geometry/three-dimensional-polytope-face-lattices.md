# Three-dimensional polytope face lattices

`polytope.face_lattice.compute` accepts a labelled rational V-representation
whose convex hull is full-dimensional in three-space. It recomputes the complete
exact facet incidence from the source vertices, identifies the extreme source
rows, and returns every face of the hull: the empty face, vertices, edges,
facets, and the whole polytope. Face records retain sorted indices into the
ordered source vertex tuple. The Hasse cover list records every incident pair
whose dimensions differ by one.

The source value can include distinct redundant generators. They remain in the
retained source, while face labels contain only its extreme vertices. Source
coordinates are limited to 32 digits per rational component. The operation
admits at most 64 source rows, 376 faces, 932 covers, one million postprocessing
work units, and two million result characters. Exact facet enumeration uses the
existing bounded double-description kernel and its ray, candidate-pair, and
coefficient-height admission.

In dimension three, every edge is the intersection of exactly two facets.
The operation derives edges by intersecting pairs of complete facet vertex
sets, after removing non-extreme source rows by the active-normal rank test.
Euler's relation and the planar graph bounds provide output limits: with `v`
extreme vertices, `e <= 3v - 6`, `f <= 2v - 4`, the face count including the
empty and whole faces is at most `6v - 8`, and the cover count is at most
`15v - 28`.

This result is a finite face poset, not a quotient by a crystallographic group
and not a cellular chain complex. The retained source is revalidated at the
operation boundary; its facet data are produced internally rather than
accepted as a caller-authored geometric claim.
