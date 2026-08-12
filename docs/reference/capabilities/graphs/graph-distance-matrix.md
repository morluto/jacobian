# Graph distance matrix

[Documentation home](../../../index.md)

`graph.distance_matrix.compute` returns one complete typed value of exact
unweighted shortest-path distances for a bounded finite simple undirected
graph. The operation uses the existing `GraphInvariantRequest` contract, so
inputs retain the established limit of 32 vertices and 496 edges.

## Result semantics

The result is bound to
`unweighted-shortest-path-distance-matrix.v3` and makes its representation
choices explicit:

- `target_vertices` is the complete input vertex set in lexicographic ascending
  order and labels every distance-vector target;
- `rows` is in the same canonical order, and every row carries its own
  `source_vertex` label next to a `distances_by_target` mapping;
- every entry is therefore bound directly to both its source and target labels
  rather than relying on detached positional indices;
- a finite entry is the number of edges in a shortest path;
- an unreachable pair is represented by JSON `null`, never a numeric sentinel;
- the diagonal is zero, finite off-diagonal entries are positive, and the
  matrix is symmetric; and
- `connected` is true exactly when the graph is nonempty and every entry is
  finite.

The empty graph returns empty targets and rows with `connected = false`. A
singleton returns one labelled row mapping itself to zero with
`connected = true`.

The typed result rejects inconsistent ordering, coverage, diagonal, symmetry,
component closure, triangle inequality, or connectedness before publication.

## Scope and composition

This operation exposes the distance matrix as one inspectable mathematical
value. It does not compute a diameter, radius, vertex eccentricity, distance
between derived vertex sets, or a conjecture-specific inequality. Agents can
compose the labelled rows with independently obtained vertex sets without
relabeling a detached positional matrix.

## Independent verification

`graph.distance_matrix.verify` is a separate checker operation. It accepts the
exact producer input and candidate value and independently replays a
breadth-first traversal from every source using only Python standard-library
adjacency sets, queues, and integer distances. It does not import NetworkX or
the producer implementation.

The checker rejects malformed metadata, source or target ordering, missing or
extra rows, entry types, diagonal values, asymmetry, incorrect edge distances,
triangle violations, finite-component closure violations, and false shortest
paths. Acceptance requires exact comparison of every finite distance and every
unreachable `null`.

Only the operator-authorized checker may emit a verification record. Failure,
rejection, timeout, cancellation, or unavailable checker execution does not
verify the candidate.
