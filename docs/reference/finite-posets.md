# Finite posets

[Documentation home](../index.md)

- Status: Experimental exact domain contract
- Provider: pinned NetworkX producer
- Independent replay: standard-library clean-process checker
- Producer assurance: at most `COMPUTED`

The `poset` bundle exposes four atomic capabilities:

- `poset.finite.materialize`;
- `poset.width.compute`;
- `poset.linear_extensions.count`; and
- `poset.mobius_function.compute`.

Each producer has a separate verifier when bundled checker authorization is
enabled: `poset.finite.verify`, `poset.width.verify`,
`poset.linear_extensions.verify`, and `poset.mobius_function.verify`. Exact
and deterministic producer output does not self-certify.

## Finite-poset materialization

`poset.finite.materialize` accepts a labelled finite carrier and one explicitly
interpreted relation:

- `COVER_EDGES` means the supplied strict pairs are exactly the Hasse edges.
  Self-pairs and transitively redundant edges are invalid.
- `COMPARABLE_PAIRS` means the supplied pairs are the complete order relation.
  The request declares whether every reflexive diagonal pair is `REQUIRED` or
  all diagonal pairs are `FORBIDDEN`.

Labels are unique bounded ASCII identifiers. Relation endpoints must lie in
the carrier. Duplicate pairs, cycles, reverse comparable pairs, an incomplete
transitive relation, and an inconsistent reflexive policy fail complete
request validation before any operation artifact is written.

The canonical finite-poset artifact contains:

- the lexicographically ordered carrier;
- every strict comparable pair;
- the transitive reduction;
- every canonical incomparable pair;
- all minimal and maximal elements;
- a content digest; and
- ranks exactly when the whole poset is graded.

For this contract, a graded poset has rank zero at every minimal element, every
cover increases rank by one, and all maximal elements have the same rank.
`ranks` is `null` for a non-graded poset; the producer never fills guessed or
component-local ranks into that field.

## Width and its dual certificate

`poset.width.compute` returns one exact width together with:

- a maximum antichain;
- a partition of the complete carrier into the same number of chains; and
- the bipartite matching that joins consecutive elements of those chains.

The antichain is a lower-bound witness. The chain partition is an upper-bound
witness. When their sizes agree, Dilworth's theorem closes the optimality
claim without trusting the producer's matching search. The checker verifies
pairwise incomparability, chain comparability, exact disjoint coverage,
matching incidence, and equality of both bounds.

## Linear-extension count

`poset.linear_extensions.count` supports at most 14 elements. It performs exact
subset dynamic programming over order ideals and returns every visited ideal
state, not only the final integer. Each state preserves:

- its bit mask in the canonical element order;
- its cardinality;
- every removable maximal element; and
- its exact recurrence count.

The result declares `ALL_ORDER_IDEALS`, reports the number of examined subsets
and ideal states, and binds the complete recurrence table with a canonical
digest. In the largest allowed antichain the table has 65,536 states. Requests
above the element bound fail before computation; no sampling or approximate
count is substituted.

The independent checker enumerates every subset again, distinguishes ideals
from non-ideals, replays every recurrence and base case, and binds the final
count to the full-carrier state.

## Möbius function

`poset.mobius_function.compute` has two non-interchangeable scopes:

- `COMPLETE_MATRIX` returns every interval \(x\leq y\); and
- `SELECTED_INTERVALS` returns exactly the requested valid intervals.

Every value includes the canonical recurrence contributions used in

\[
\mu(x,x)=1,\qquad
\mu(x,y)=-\sum_{x\leq z<y}\mu(x,z).
\]

The checker reconstructs the partial order and replays the equivalent
convolution identity

\[
\sum_{x\leq z\leq y}\mu(x,z)=\delta_{xy}.
\]

A selected interval result never claims matrix completeness, and a complete
matrix request cannot carry an interval selection.

## Bounds and public cases

Version 1 uses these fail-closed limits:

| Quantity | Limit |
| --- | ---: |
| Poset carrier | 64 elements |
| Presented or materialized relation | 4,096 pairs |
| Linear-extension carrier | 14 elements |
| Linear-extension recurrence table | 65,536 ideal states |

The regression suite covers the empty poset, singleton, chains, antichains,
the diamond, non-graded examples, complete and selected Möbius scopes,
relabeling, cycles, redundant covers, incomplete comparable relations, and
forged certificate payloads. A separate development audit exhaustively
cross-checked all 1,100 fixed-topological-order presentations through order
five against independent replay, including exact topological-sort counts.

## Trust boundary

NetworkX supplies maintained DAG closure, reduction, lexicographic topological
sorting, and bipartite matching primitives to the producer. The independent
checker imports neither NetworkX nor poset contracts or producer code. It
reconstructs closure and reduction with standard-library traversal, checks
the dual width witnesses directly, rebuilds every ideal recurrence, and
replays every requested Möbius convolution.

Replay binds the exact input artifact, result artifact, poset semantics,
candidate digest, witness format, and operator-authorized checker identity.
Malformed, rejected, interrupted, or unsupported replay creates no
verification record and makes no opposite mathematical conclusion.

Infinite orders, approximate extension counting, order dimension, unlabeled
isomorphism, lattice operations, matroids, and hypergraphs remain outside this
contract.
