# Graph diameter and radius verification

[Documentation home](../../../index.md)

`graph.invariant.diameter.compute` and
`graph.invariant.radius.compute` version `2` retain their existing producer
contracts and `COMPUTED` assurance. Operator-authorized
`graph.invariant.diameter.verify` and `graph.invariant.radius.verify`
operations can promote one exact submitted `{input, candidate}` claim to `VERIFIED` after
independent all-sources breadth-first replay.

## Exact claims and conventions

Each verifier checks one complete typed candidate against the exact submitted finite
simple undirected graph:

- for a nonempty connected graph, diameter is the maximum vertex eccentricity
  and radius is the minimum vertex eccentricity;
- a singleton graph has diameter and radius zero; and
- an empty or disconnected graph has no numeric value under these contracts
  and returns `status = NOT_APPLICABLE`, `connected = false`, and
  `exactness = NOT_APPLICABLE`.

The claim is bound to canonical input and candidate digests, graph semantics,
checker identity, checker source digest, and executable identity.
The verifier does not certify a directly supplied graph, an unbounded graph
family, or a theorem that later uses the metric.

## Independent replay

The producers use NetworkX. The checker is separately implemented with Python
standard-library adjacency sets, queues, and integer distances. It imports
neither NetworkX nor the producer package.

For every source vertex, the checker runs breadth-first traversal, rejects a
connected claim if any vertex is unreachable, and records the maximum finite
distance as that source's eccentricity. It then independently takes the
maximum for diameter or minimum for radius. The producer and checker share only
passive artifact-envelope parsing.

## Verification obligation ledger

| Obligation | Independent replay | Failure meaning |
| --- | --- | --- |
| Artifact binding | Recompute and compare claim, semantics, candidate, lineage, witness-envelope, and payload digests. | Reject this evidence; no mathematical conclusion. |
| Graph validity | Parse the complete graph and reject malformed vertices, loops, duplicate undirected edges, undeclared endpoints, and order above 32. | Reject malformed or unsupported evidence. |
| Connectivity convention | Exhaust traversal from every source, with empty and disconnected graphs mapped only to the contract's inapplicable result. | Reject a mismatched status or numeric value. |
| Exact metric | Recompute every eccentricity and its exact maximum or minimum. | Reject a forged diameter or radius. |
| Result normalization | Require the exact result fields and consistent status, value, connectivity, exactness, and detail. | Reject noncanonical or internally inconsistent evidence. |
| Authorization and runtime | Dispatch only the operator-authorized checker matching the schemas, semantics, format, source digest, and executable identity. | Unavailable, timeout, cancellation, or error remains non-conclusive. |

Acceptance reports exhaustive finite replay. A rejection means only that the
submitted result was not verified; it never establishes another diameter,
radius, or connectivity claim.
