# Cubical cell vertices

`topology.cubical.cell.vertices.compute` returns the complete lattice-vertex
set of one elementary cubical cell. A cell with `k` unit intervals has
`2^k` vertices; every degenerate interval contributes its fixed coordinate and
does not increase the count. Each output vertex is a zero-dimensional
`CubicalCell`, preserving the source's ordered ambient axes. Vertices are
returned in lexicographic coordinate order.

The operation admits the source axis count, the `2^k` output count, coordinate
digits, and a conservative result-size bound before enumerating vertices. It
accepts coordinates of at most 64 decimal digits and limits the encoded result
to 512 KiB. A source cell that is already a point returns that same point as its
single vertex.

The vertex convention follows the elementary integer-lattice cell model used
throughout the cubical-complex operations. This operation does not close a
family of cells under faces or construct a cubical complex.
