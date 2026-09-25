# Linear matroid optimization

[Documentation home](../../../index.md) · [Operation references](../index.md)

These operations consume a canonical `LinearMatroid` represented by the
columns of a prime-field matrix. Weights are exact integers attached to those
columns by a `MatroidWeightFunction` whose explicit ground axis must equal the
matroid's axis, so the table cannot be detached or silently permuted.

## Graphic matroids

`matroid.graphic.represent.compute` converts a finite simple undirected graph
to its incidence-column representation over `GF(2)`. Rows use lexicographically
sorted graph vertices; columns use sorted endpoint pairs. The ground label of
each column is the compact JSON encoding of that endpoint pair, so labels with
punctuation or Unicode remain unambiguous. An edge set is independent exactly
when it is a forest. For a disconnected graph, every basis is a spanning
forest, with rank `|V| - c(G)` including isolated vertices.

The result is an ordinary `LinearMatroid` and composes directly with keyed
maximum-weight basis and independent-set operations. The constructor admits at
most 256 vertices and 256 edges before allocating its incidence matrix, and
bounds the retained endpoint-label axis at 65,536 Unicode codepoints counted
once per edge endpoint. A deserialized matrix remains an ordinary represented
matroid; consumers that need graph provenance should retain the original graph
and its edge-label map.

## Maximum-weight independent set

`matroid.independent_set.maximum_weight.compute` returns one maximum-weight
independent subset. It considers only positive-weight elements, ordered by
decreasing weight and then increasing ground index, and greedily retains an
element exactly when it raises the selected column rank. This is the matroid
greedy algorithm for the independent-set polytope.

The result retains the source matroid and weights, the selected indices, their
exact total weight and rank, and the complete positive-weight scan order. The
empty set is valid; zero and negative weights are never forced into the result.
This differs from `matroid.basis.maximum_weight.compute`, which must return a
basis even when some basis elements have negative weights.

The current exact envelope admits at most 256 ground elements, 256 matrix
rows, fewer than 12 decimal digits per weight, and a retained ground axis of
at most 65,536 Unicode codepoints. Before rank calculations, the operation
charges the number of positive-weight rank probes at the full
representation-rank cost, plus the final selected-set rank and the retained
index, weight-digit, and duplicated-axis label allocation, against the
50,000,000-unit matroid work and result-output limits.

The returned value can be independently replayed with
`verify_maximum_weight_independent_set`. That replay recomputes the deterministic
greedy result against the retained matrix and weights; a feasible but
suboptimal subset does not verify.

## Maximum-cardinality intersection

`matroid.intersection.compute` returns one exact maximum-cardinality common
independent set `I` of two represented matroids on the same labelled ground,
with an Edmonds min-max witness
`r₁(A) + r₂(E \\ A) = |I|`. The result retains the exact ranks of `I` in
both sources as `rank_first_common` and `rank_second_common`; each must equal
`cardinality`. These rank fields extend the serialized result schema while the
existing operation ID and `{first, second}` request schema remain unchanged.

Deserializing a result checks its shape and the equality of those rank fields
with the cardinality without replaying matrix computations. Consumers that rely
on the retained rank claims can call `replay_intersection_result`, which
recomputes both common-set ranks and both min-max witness ranks against the
retained matrices. The returned common set and matching min-max upper bound
establish global maximality independently of the augmenting-path search.

The request admits at most 256 ground elements and a derived 50,000,000-unit
work envelope covering source ranks, exchange probes, common-set and witness
rank checks, and result materialization. Requests outside this envelope fail
admission; they do not establish a smaller maximum.

## Common basis of two matroids

`matroid.intersection.common_basis.compute` returns the exact maximum common
independent set, each source matroid's full rank, and an Edmonds min-max witness
`r₁(A) + r₂(E \\ A) = |I|`. `COMMON_BASIS` is returned only when the maximum set
has cardinality equal to both source ranks. `NO_COMMON_BASIS` includes a reason:
`SOURCE_RANK_MISMATCH` when source ranks differ, or
`MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL` when equal source ranks exceed the
maximum intersection cardinality. The latter decision is backed by the exact
min-max witness.

The result is bound to both represented matroids. Deserialization checks the
shape and consistency of the returned ranks, status, reason, basis, and witness
without repeating matrix computations. `verify_common_basis_result` replays
the source ranks, feasibility ranks, and witness ranks against the retained
matrices, including for values constructed outside the wire path. The admitted
ground and work limits match maximum-cardinality matroid intersection:
admission precomputes both source ranks, reuses them for the closed decision,
bounds each retained ground axis at 65,536 Unicode codepoints, and charges the
exchange search at the regime those ranks make reachable — at most
`min(r₁, r₂) + 2` searches of cached probes on matrices of at most
`min(r₁, r₂) + 1` columns. A rank-zero
source is presolved exactly. Requests beyond the exact envelope fail
admission; they do not produce a negative common-basis conclusion.

## Supplied weighted-intersection certificate

`matroid.intersection.weighted_certificate.check` checks a caller-supplied
common independent set `I` and an integral split `w = u + v` on the exact
shared labelled ground. It recomputes the exact maximum-weight independent-set
value for `u` in the first linear matroid and for `v` in the second using the
single-matroid greedy operations above. It returns a source-bound result only
when `I` is independent in both sources and those two maxima sum to `w(I)`.
For every common independent set `J`,

\[
w(J)=u(J)+v(J)\leq \max_{K\in\mathcal I_1}u(K)+\max_{L\in\mathcal I_2}v(L)=w(I),
\]

so the returned candidate is maximum-weight, including when all input weights
are negative and the empty set is optimal. The general integral
weight-splitting criterion follows from box total dual integrality of the
common-independent-set polytope; see Schrijver, *Combinatorial Optimization*,
Chapter 41, §41.4, Theorem 41.12. Frank's
[weight-splitting proof](https://egres.elte.hu/qp/egresqp-08-03.pdf) treats the
common-basis (equal-rank) setting, so it is not the sole basis for this
arbitrary common-independent-set contract.

The request and result retain the original weights, both split-weight
single-matroid maximizers, and the candidate. The ground and row limits are
256; each weight entry follows the existing fewer-than-12-decimal-digit
contract. One aggregate 50,000,000-unit envelope covers both greedy scans and
both candidate feasibility ranks; each source also carries the shared
65,536-codepoint bound on its retained ground axis, and each greedy phase
must fit the single-matroid index, weight-digit, and duplicated-axis output
budget before any rank expansion.
A serialized result can be independently recomputed with
`verify_weighted_intersection_result`.
This operation checks a supplied split; it does not find an optimum candidate
or construct the split.
