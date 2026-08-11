# Maximum-matching certificate and verification

[Documentation home](../../../index.md)

`graph.invariant.maximum_matching.compute` version `3` returns a feasible
matching together with a Tutte–Berge barrier certificate. The producer remains
`COMPUTED`. An operator-authorized
`graph.invariant.maximum_matching.verify` capability may promote the exact
stored result to `VERIFIED` only after independent clean-process replay.

## Exact claim and scope

The verifier checks one claim:

> The stored witness edges form a maximum-cardinality matching of the exact
> stored finite simple undirected graph.

The claim is bound to the producer input artifact, result artifact, graph
semantics, result schema version, checker identity, and checker provider
runtime. It does not certify a graph supplied directly by the caller, a
different matching, or any theorem that uses the matching as an intermediate
fact.

The producer accepts at most 64 vertices and 2,016 edges through a
matching-specific graph contract. Other graph invariants retain their own
existing bounds; this does not broaden coloring or exponential graph-search
operations. The independent checker uses only bounded standard-library graph
traversal. It imports neither
NetworkX, which constructs the matching and certificate, nor Python-FLINT,
which serves unrelated polynomial and matrix checkers.

## Producer certificate

The result contains:

- a canonical, vertex-disjoint edge witness;
- its declared cardinality; and
- a `TUTTE_BERGE_BARRIER` certificate containing a canonical vertex set
  \(U\), the declared number of odd components of \(G-U\), and the resulting
  upper bound.

The NetworkX producer obtains a barrier through the Gallai–Edmonds
decomposition. It computes

\[
D=\{v:\nu(G-v)=\nu(G)\},\qquad U=N(D)\setminus D.
\]

Certificate construction is untrusted. A failure to construct a valid
certificate is an operation failure, not evidence that the matching is
nonmaximum.

## Verification obligation ledger

| Obligation | Independent replay | Failure meaning |
| --- | --- | --- |
| Exact artifact binding | Recompute and compare claim, semantics, candidate, lineage, witness-envelope, and payload digests. | Reject this evidence; no mathematical conclusion. |
| Graph validity | Parse the complete bounded simple graph; reject loops, duplicate undirected edges, undeclared endpoints, malformed vertices, and unsupported order. | Reject malformed or unsupported evidence. |
| Matching feasibility | Require canonical witness edges, graph membership, vertex disjointness, and equality between witness length and declared cardinality. | Reject the candidate; no opposite claim follows. |
| Certificate scope | Require one canonical barrier subset of the exact graph and the exact versioned certificate shape. | Reject the certificate. |
| Odd-component count | Remove the barrier and independently enumerate connected components by breadth-first traversal; count components of odd order. | Reject a mismatched certificate. |
| Tutte–Berge upper bound | Recompute \(b=(|V|+|U|-q(G-U))/2\), require integral parity, and require the declared upper bound and matching cardinality to equal \(b\). | Reject the optimality claim. |
| Optimality conclusion | Combine the feasible matching lower bound with the independently checked Tutte–Berge upper bound. | Only equality permits acceptance. |
| Authorization and runtime | Dispatch only the operator-authorized checker whose source digest, schemas, semantics, evidence format, and provider runtime match the record. | Unavailable, timeout, cancellation, or error remains non-conclusive. |

The Tutte–Berge theorem gives

\[
\nu(G)=\frac{1}{2}\min_{U\subseteq V}
\left(|V|+|U|-q(G-U)\right).
\]

Consequently any checked barrier supplies an upper bound, while the checked
matching witness supplies a lower bound. Equality discharges the maximum
cardinality obligation without replaying the producer's matching algorithm.

## Adversarial matrix

The checker tests include:

- a feasible but nonmaximum matching;
- a witness edge absent from the source graph;
- intersecting, duplicated, reversed, or noncanonical witness edges;
- a barrier vertex outside the source graph;
- a mutated barrier, odd-component count, or upper bound;
- source, candidate, semantics, claim-binding, or payload-digest substitution;
- a valid certificate rebound to another graph or result; and
- checker unavailability, timeout, cancellation, and unauthorized installation.

Every rejection returns `UNKNOWN`; it never asserts that another matching is
maximum.
