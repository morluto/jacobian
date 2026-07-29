# Exact planar geometry

[Documentation home](../index.md)

- Status: Current implementation reference; contracts are experimental
- Domain: `geometry`
- Producer backend: pinned SymPy exact rational geometry
- Checker backend: Python standard-library `Fraction`, isolated from SymPy

The planar-geometry bundle exposes atomic outcomes over canonical rational
coordinates. Inputs are bounded to at most 128 distinct polygon or point-set
vertices and are fully validated before computation or artifact writes.

## Closed segment intersection

`geometry.segments.intersection.compute` accepts two closed segments. Equal
endpoints are legal and explicitly denote a point segment. Its discriminated
result preserves exactly one of:

| Status | Data |
| --- | --- |
| `DISJOINT` | No intersection data |
| `POINT` | Exact point plus `PROPER`, `ENDPOINT_TOUCH`, or `DEGENERATE_TOUCH` |
| `OVERLAP` | The maximal common closed segment in canonical endpoint order |

The result never collapses a collinear overlap or a degenerate touch into an
untyped boolean.

The older `geometry.segment.compute.midpoint` ID remains unchanged. The
singular/plural difference is retained for compatibility: this addition does
not rename or alias existing capability IDs.

## Polygon decisions

`geometry.polygon.simple.decide` accepts an open-ring representation: at least
three unique vertices, without repeating the first vertex at the end. It checks
every unordered edge pair in deterministic lexicographic index order. A
positive result records the complete pair count. A negative result preserves
the first violating pair and its exact point or overlap intersection.

Adjacent edges are valid only when their intersection is exactly their shared
endpoint. Non-adjacent edges must be disjoint. Consequently the decision
detects proper self-crossings, self-touches, and adjacent collinear overlap.
Duplicate vertices and zero-length ring edges are rejected structurally before
the producer runs.

`geometry.polygon.point.classify` returns `INSIDE`, `BOUNDARY`, or `OUTSIDE`.
Its request is rejected before computation unless the supplied ring is simple.
A boundary result identifies the exact edge containing the point. Reversing
the ring orientation does not change the classification.

## Convex hull normalization

`geometry.points.compute.convex_hull` now emits a canonical candidate:

- a point hull contains that point;
- a collinear hull contains its two lexicographically ordered extremes; and
- a polygon hull is strictly counterclockwise and starts at its least vertex.

This normalization closes the previous checker gap without changing the
capability ID.

## Independent verification

`geometry.result.verify` now supports the three operations above, convex hull,
and the previously supported distance, midpoint, orientation, and centroid
operations. The operator-authorized checker runs in a clean process and
imports neither SymPy nor geometry producer modules.

For segment and polygon results it independently replays rational
determinants, interval containment, all required edge pairs, exact ray
crossings, and boundary tests. Convex hull replay uses an independent monotone
chain implementation. Verification binds the exact input artifact, result
artifact, semantics, candidate digest, and witness envelope. Producers remain
`COMPUTED`; only an accepted bound checker run returns `VERIFIED`.

Delaunay/Voronoi construction, arbitrary algebraic coordinates, three-
dimensional geometry, and H/V polyhedral conversion remain separate provider
or capability gates.
