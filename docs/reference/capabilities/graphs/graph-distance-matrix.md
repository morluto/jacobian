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

`graph.distance_matrix.verify` is a separate checker operation. It consumes the
exact producer input plus one complete typed candidate inline and can promote
that matrix to `VERIFIED`. The operator-authorized checker independently
replays a breadth-first traversal from every source using only Python
standard-library adjacency sets, queues, and integer distances. It does not
import NetworkX or the producer package.

The checker rejects malformed metadata, source or target ordering, missing or
extra rows, entry types, diagonal values, asymmetry, incorrect edge distances,
triangle violations, finite-component closure violations, and false shortest
paths. Acceptance requires exact comparison of every finite distance and every
unreachable `null`.

Only the operator-authorized checker may emit a verification record. The
verification record is bound to the canonical graph input and matrix candidate
digests, semantics, checker source digest, and provider runtime. Rejection,
timeout, cancellation, unavailable runtime, or checker error remains `UNKNOWN`
and cannot produce `VERIFIED`.

## Public composition evidence

The frozen public matched evaluation in
[Capability workflow evaluations](../../evaluations/benchmark-contracts.md#task-and-verifier-validation)
used three control/treatment pairs. All treatments autonomously discovered the
producer and verifier, preserved independently replayable matrix evidence, and
correctly derived a restricted-set distance profile without substituting
diameter, radius, or eccentricity. Only the exact matrix received a bound
verification record; the derived profile did not.

This public answer-visible result is regression evidence for composition, not
a broad portfolio-value claim. It does not justify adding a restricted-set
distance capability by itself.
