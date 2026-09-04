# Exact planar geometry

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Planar geometry operations accept bounded exact rational points, segments,
lines, triangles, and polygons. The catalog includes direct computations for
intersection, projection, midpoint, orientation, centroid, circumcircle,
convex hull, signed area, triangulation, and projective line-arrangement flats,
as well as direct predicates for parallelism, perpendicularity, collinearity,
concyclicity, and polygon simplicity.

Inputs are validated as their owning geometric value before computation. Results
are returned inline and can be passed to another compatible typed operation;
there is no geometry artifact or generic verification service.

## Point-configuration orbit profiles

`geometry.point_configuration.euclidean_orbit_profile.compute` canonicalizes
one bounded labelled rational point configuration. It returns the
lexicographically least complete squared-distance matrix over source
relabelings, retaining a deterministic source-to-canonical map. The isometry
form uses the source distances unchanged; the similarity form divides them by
the least positive squared distance. Equal forms characterize congruence and,
after the common positive normalization, similarity, respectively, including
reflections.

Admission separately bounds pairwise exact-distance output, rational height,
and the complete exhaustive-permutation replay. The current release admits
configurations whose complete relabeling search needs at most 8! permutations;
larger finite sources reject before canonicalization rather than returning an
unproved canonical form.

## Convex-polygon triangulation

`geometry.polygon.triangulation.minimum_weight.compute` minimizes caller-supplied
exact rational non-hull diagonal weights. Its recurrence charges a selected
diagonal exactly once, when it is the boundary of its non-root subproblem.

`geometry.polygon.triangulation.minimum_euclidean_weight.compute` instead owns
the non-hull Euclidean-length objective for one strict counter-clockwise convex
rational polygon with 4 to 68 vertices. Admission bounds the dynamic-program
work, exact expression terms, and retained diagonal cardinality directly. A successful result
retains every selected diagonal's exact rational
squared length and represents the source-bound optimum exactly as the ordered
sum of their positive square roots. It never labels a decimal approximation as
exact.

The Euclidean dynamic program uses pinned Arb at 128 bits only to decide the
finite comparisons among those exact expressions. An outward-rounded interval
that excludes zero certifies one strict order; syntactically identical exact
expressions are deterministic ties. If a different pair overlaps zero, the
operation returns `COMPARISON_UNRESOLVED` with the two exact expressions and
does not claim a triangulation or optimum. This is an operation-specific
incomplete mathematical result, not a generic verification framework.
