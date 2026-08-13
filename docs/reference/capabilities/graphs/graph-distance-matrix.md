# Graph distance matrix

[Documentation home](../../../index.md)

`graph.distance_matrix.compute` returns one complete matrix of exact
unweighted shortest-path distances for a bounded finite simple undirected
graph. The producer has a distance-matrix-owned polynomial-time contract with
a limit of 64 vertices and 2,016 edges. This dedicated boundary does not widen
the shared 32-vertex graph contract used by NP-hard coloring and optimization
operations or the 18-vertex Hamiltonian-path search boundary.

## Result semantics

The result is bound to
`unweighted-shortest-path-distance-matrix.v1` and makes its representation
choices explicit:

- `vertices` is the complete input vertex set in lexicographic ascending order;
- `distances[i][j]` covers every ordered pair of vertices in that order;
- a finite entry is the number of edges in a shortest path;
- an unreachable pair is represented by JSON `null`, never a numeric sentinel;
- the diagonal is zero, finite off-diagonal entries are positive, and the
  matrix is symmetric; and
- `connected` is true exactly when the graph is nonempty and every matrix
  entry is finite.

The empty graph returns an empty matrix with `connected = false`. A singleton
returns `[[0]]` with `connected = true`.

The result model rejects inconsistent ordering, shape, diagonal, symmetry,
component closure, triangle inequality, or connectedness before an artifact is
written.

## Scope and composition

This capability exposes the distance matrix as one inspectable mathematical
outcome. It does not compute a diameter, radius, vertex eccentricity, distance
between derived vertex sets, or a conjecture-specific inequality. An agent can
derive such quantities by composing this artifact with independently obtained
sets or other capabilities.

The producer is capped at `COMPUTED`. That assurance means the bounded
operation completed and produced a typed artifact; it is not independent
verification of the distance claim.

## Independent verification

`graph.distance_matrix.verify` consumes the exact producer input plus one complete
typed candidate inline and can promote that matrix to `VERIFIED`. The
operator-authorized checker uses
only Python standard-library adjacency sets, queues, and integer distances. It
does not import NetworkX or the producer package.

The checker first rejects malformed metadata, ordering, shape, entry types,
diagonal values, asymmetry, incorrect edge distances, triangle violations, and
finite-component closure violations. Those conditions are only fast rejection
checks. Acceptance still requires an exhaustive breadth-first traversal from
every source and exact comparison of every finite distance and unreachable
`null`.

The verification record is bound to the canonical graph input and matrix
candidate digests, semantics, checker source digest, and provider runtime.
Rejection, timeout, cancellation, unavailable runtime, or
checker error remains `UNKNOWN` and cannot produce `VERIFIED`.

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
