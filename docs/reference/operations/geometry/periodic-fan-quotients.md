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

For polygons, strict convexity uses the exact global supporting-line test —
every directed boundary edge must have all other listed vertices strictly to its
left, which also rejects star-ordered self-intersecting cycles whose local turn
signs are all positive — and coverage uses exact shoelace volumes. Intersection
and common-face checks use exact rational feasibility. Quotient construction
enumerates only actual polygon faces. Work is bounded by the existing
presentation limits, a 12-vertex polygon cap, and the feasibility tableau row
cap, whose square is the per-step generated-row bound; admission preflights the
structural initial-tableau bound of every feasibility problem, so the
generated-row bound needs no mid-computation rejection. Translation stabilizers
test at most one candidate per vertex, avoiding factorial permutation
enumeration.
