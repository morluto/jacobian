# Cubical standard triangulation

`topology.cubical_complex.standard_triangulation.compute` converts a finite
integer-lattice cubical complex to Jacobian's canonical
`FiniteSimplicialComplex` value using the Freudenthal (staircase) triangulation.
For a `d`-cube, each permutation of the active coordinate axes gives one path
simplex from the lower vertex to the upper vertex. The standard construction
partitions the unit `d`-cube into `d!` such simplices; its restrictions to cube
faces use the same coordinate order, so adjacent source cubes induce the same
triangulation on their shared face. This follows directly from the fixed ambient
coordinate order in the path-simplex construction of Freudenthal,
[*Simplizialzerlegungen von beschränkter Flachheit*](https://doi.org/10.2307/1968813),
*Annals of Mathematics* 43 (1942), 580–582.

The result carries the complete face-closed cubical source, the canonical
finite simplicial target, the target-vertex to exact lattice-coordinate map,
and each inclusion-maximal cubical source cell's simplex family. Vertex labels
are local canonical indices; use `vertex_map` to recover coordinates.
Automatic result construction and JSON decoding check bounded typed axes. A
consumer that relies on the authored source/target relation calls
`require_valid_transport()`. This explicit check admits the source size,
coordinate digits, simplex count, vertex work, and factorial-weighted face work
before rebuilding the Freudenthal relation.

Admission precedes simplex and face expansion. It bounds source presentation
size and maximal-cell comparison work; target dimension is at most 7, target
vertices at most 64, target facets at most 128, and target faces at most 2,048.
The operation also admits a factorial-weighted simplex-face candidate bound
before enumerating permutations or simplex faces. Larger triangulations need a
separately reviewed target simplicial carrier with larger published limits.

The returned value supports existing simplicial operations directly. This
operation does not return cubical-to-simplicial chain maps or assert a chain
equivalence; those remain separate work.
