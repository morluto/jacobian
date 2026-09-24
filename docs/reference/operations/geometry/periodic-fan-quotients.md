# Periodic fan quotient validation

`periodic_fan.quotient.validate` validates a bounded lattice-periodic cell
presentation and returns its exact quotient face incidence, face-orbit map,
period index, and translation stabilizers. The validated finite cells represent
all their lattice translates; the kernel never materializes the infinite set.

Maximal cells may be simplices in ranks one through four. The additional
non-simplicial case is a strictly convex rank-two polygon with at most 12
vertices. Its vertex indices are listed counterclockwise around the boundary,
beginning with the smallest index. Its faces are its vertices, boundary edges,
and the polygon itself; diagonals are not faces. Cells in rank three and four
remain simplicial. Unimodularity claims apply to simplex cells only.

For polygons, strict convexity and coverage use exact integer orientation and
shoelace tests. Intersection and common-face checks use exact rational
feasibility. Quotient construction enumerates only actual polygon faces. Work is
bounded by the existing presentation limits, a 12-vertex polygon cap, and the
feasibility tableau row cap. Translation stabilizers test at most one candidate
per vertex, avoiding factorial permutation enumeration.
