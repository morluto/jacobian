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
most 256 vertices and 256 edges before allocating its incidence matrix. A
deserialized matrix remains an ordinary represented matroid; consumers that
need graph provenance should retain the original graph and its edge-label map.

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
rows, and fewer than 12 decimal digits per weight. Before rank calculations,
the operation charges the number of positive-weight rank probes at the full
representation-rank cost, plus the final selected-set rank and output size,
against the 50,000,000-unit matroid work limit.

The returned value can be independently replayed with
`verify_maximum_weight_independent_set`. That replay recomputes the deterministic
greedy result against the retained matrix and weights; a feasible but
suboptimal subset does not verify.

## Common basis of two matroids

`matroid.intersection.common_basis.compute` returns the exact maximum common
independent set, each source matroid's full rank, and an Edmonds min-max witness
`r₁(A) + r₂(E \\ A) = |I|`. `COMMON_BASIS` is returned only when the maximum set
has cardinality equal to both source ranks. `NO_COMMON_BASIS` includes a reason:
`SOURCE_RANK_MISMATCH` when source ranks differ, or
`MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL` when equal source ranks exceed the
maximum intersection cardinality. The latter decision is backed by the exact
min-max witness.

The result is bound to both represented matroids. Deserializing it replays the
source ranks, feasibility ranks, and witness ranks against those matrices;
`verify_common_basis_result` independently checks the same claims for values
constructed outside the wire path. The admitted ground and work limits match
maximum-cardinality matroid intersection and include the two added full-source
rank calculations. Requests beyond the exact envelope fail admission; they do
not produce a negative common-basis conclusion.

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

so the returned candidate is maximum-weight. This is the integral
weight-splitting optimality criterion for matroid intersection; see Frank's
[weight-splitting proof](https://egres.elte.hu/qp/egresqp-08-03.pdf) and the
matroid-intersection polytope's total-dual-integrality statement in the
[UIUC combinatorial optimization notes](https://courses.physics.illinois.edu/cs586/sp2022/main.pdf).

The request and result retain the original weights, both split-weight
single-matroid maximizers, and the candidate. The ground and row limits are
256; each weight entry follows the existing fewer-than-12-decimal-digit
contract. One aggregate 50,000,000-unit envelope covers both greedy scans and
both candidate feasibility ranks; the source-bound result also has a
conservative 8 MiB serialized-size bound that includes repeated axis labels.
A serialized result can be independently recomputed with
`verify_weighted_intersection_result`.
This operation checks a supplied split; it does not find an optimum candidate
or construct the split.
