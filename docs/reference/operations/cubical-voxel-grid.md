# Construct a cubical complex from 3D binary voxels

`topology.cubical_complex.from_binary_voxels_3d.compute` accepts a rectangular
Boolean array indexed as `voxels[z][y][x]`. Coordinates in the canonical
`CubicalCell` value are ordered `(x, y, z)`, and a true entry at `(x,y,z)` is
the closed unit cube

```text
[x, x+1] × [y, y+1] × [z, z+1].
```

The result is the complete face closure of all occupied cubes, represented by
the existing `CubicalComplex`. Shared faces appear once. An all-false grid
returns `CubicalComplex(ambient_dimension=3, cells=())`, retaining the three
coordinate axes. The empty value uses the same canonical carrier as other
cubical operations.

Each side is at most 64 and the complete input grid at most `64³` Boolean
entries. The kernel validates shape and strict Boolean values while counting
occupied voxels in one accounted pass, including for typed values created
without normal model validation. It then admits at most 200,000 possible
cubical-face insertions, a maximum result of `MAX_FACE_CELLS`, and the face
generation, deduplication, canonical sort, and value-check work before
generating face keys. A single occupied voxel produces 27 cells;
a full `18 × 18 × 18` block produces 50,653 distinct cells and fits the
canonical result carrier. Larger grids are admitted or refused from the
actual occupancy and exact distinct face count, rather than the grid volume
alone.

The construction is an exact finite set transform and does not infer a
geometric realization or identify nonincident cells. The returned complex can
be consumed directly by the existing cubical f-vector, chain-complex, product,
and triangulation operations.
