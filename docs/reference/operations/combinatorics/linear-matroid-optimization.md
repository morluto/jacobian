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
256; objective entries follow the fewer-than-12-digit contract, while split
entries allow fewer than 15 digits. One aggregate 50,000,000-unit envelope
covers both greedy scans and both candidate feasibility ranks; each source
also carries the shared 65,536-codepoint bound on its retained ground axis, and
each greedy phase must fit the single-matroid index, weight-digit, and
duplicated-axis output budget before any rank expansion.
A serialized result can be independently recomputed with
`verify_weighted_intersection_result`.
This operation checks a supplied split; it does not find an optimum candidate
or construct the split.

## Computed maximum-weight intersection

`matroid.intersection.maximum_weight.compute` computes an optimum and returns
the same source-bound result type as the supplied split checker. After the
augmenting optimizer selects its best common independent set, the operation
builds its final exchange inequalities and solves their integer difference
constraints with Bellman–Ford. The resulting split makes the selected set a
maximum-weight independent set in each source separately; the two exact
greedy maximizers are retained so `verify_weighted_intersection_result` can
replay the witness. This constructs the split witness, not the rank-multiplier
chains accepted by the separate rank-dual checker.

The witness phase admits at most a quadratic number of additional exchange
rank probes and a cubic number of exact Bellman–Ford relaxations. A simple
shortest path bounds each split coordinate by `(2n+1)W`, where `n` is the
ground size and `W` is the largest absolute input weight. Requests whose
conservative split bound exceeds the fewer-than-15-digit split-witness
envelope fail admission before rank expansion; caller objective weights keep
the existing fewer-than-12-digit limit. Its aggregate work and output
bounds include witness synthesis, both greedy scans, and the retained source
matrices and axes.

## Supplied rank-dual certificate

`matroid.intersection.weighted_rank_certificate.check` checks a candidate and
two finite chains of nonnegative integer multipliers on the source rank
inequalities. For a first-side term `(S, y_S)` and second-side term `(T, z_T)`,
the checker recomputes `r_1(S)` and `r_2(T)`, verifies

\[
\sum_{S\ni e} y_S + \sum_{T\ni e} z_T \geq w_e
\quad\text{for every ground element }e,
\]

and requires

\[
\sum_S y_S r_1(S) + \sum_T z_T r_2(T) = w(I).
\]

The candidate must be independent in both sources. For every common
independent `J`, its incidence vector satisfies both matroid rank systems, so
the dual cover bounds `w(J)` by the displayed dual value. Equality with the
feasible candidate therefore proves maximum weight, including signed weights
and the empty optimum. This is the LP dual of Edmonds' common-independent-set
polytope, whose rank inequalities form a TDI system; see Schrijver,
*Combinatorial Optimization*, Chapter 41, §41.4, Theorem 41.12. The theorem's
dual may be taken with integral multipliers for integer weights. The nested
chain normal form follows by solving each side's rank dual as the ordinary
maximum-weight independent-set dual; the union of two chains has a totally
unimodular incidence matrix. See also Goemans, *Topics in Combinatorial
Optimization*, [Lecture 13](https://ocw.mit.edu/courses/18-997-topics-in-combinatorial-optimization-spring-2004/198d8ca054827e34f49f5c781fc23226_co_lec13.pdf),
which spells out this dual and chain reduction.

Each side supplies at most `n` nonempty nested subsets, for at most `2n` terms
overall. Multiplier entries are exact integers bounded by the objective and
ground limits; zero-rank terms are supported for loop subsets and have zero
dual cost. One aggregate 50,000,000-unit envelope covers every listed rank,
both candidate feasibility ranks, one bounded source-prime check, and the
elementwise coverage scan. The retained sources and derived split axes share
the 65,536-codepoint retained-axis bound, so the result is rejected before any
rank expansion when its label allocation cannot fit.

The result retains the source parents, candidate, rank terms, and a derived
integral split `w=u+v`. The split is clipped to `0≤u_e≤max(0,w_e)`, so it stays
within the existing weight-entry bound and composes with
`matroid.intersection.weighted_certificate.check`. This operation checks a
supplied rank dual; it does not find a candidate or construct the dual. The
weighted optimizer is a separate operation.

## Maximum-weight common independent set

`matroid.intersection.maximum_weight.compute` computes one exact maximum-
weight common independent set for two represented matroids on the same labelled
ground and prime field. The empty set is among the candidates, so the result
may be empty when all feasible nonempty sets have nonpositive weight. When
optimal sets have different cardinalities, the result uses the smaller one;
otherwise it returns the candidate found by the deterministic exchange search.

The kernel is Frank's weight-splitting augmenting algorithm. At cardinality
`k`, it maintains a common independent set optimal at that cardinality and an
integral split `c₁+c₂=w`. Tight exchange arcs give the shortest augmentation.
When none exists, a positive minimum slack adjusts the split on the reachable
vertices and the exchange search repeats. An infinite slack proves there is no
larger common independent set; the operation then chooses the largest objective
among the cardinality-optimal sets it computed. The split maintained by the
augmenting search is private algorithm state and is not returned directly;
after selecting its candidate the operation synthesizes and returns the
replayable exchange-split witness described in the Computed maximum-weight
intersection section above. Use one of the
separate certificate-checking operations when a caller supplies an authored
optimality witness. The algorithm and its correctness argument are in
[Schrijver and Korte–Vygen, §13.7](https://www.mathematik.uni-muenchen.de/~kpanagio/KombOpt/book.pdf).

The request admits at most 256 ground elements, the existing 11-decimal-digit
weight limit, and the shared 65,536-codepoint retained ground-axis bound.
Admission accounts for the
exchange rank calls, selected-matrix copies and residue validation, one
bounded source-prime check, every reachable-graph and slack scan, and the
maximum integer width of the private split before the first rank expansion. With `n`
elements, `rᵢ=max(1, rowsᵢ)`, `κᵢ=rᵢ n min(rᵢ,n)`, and
`q=n(n+1)²+1`, the rank work bound is `q(κ₁+κ₂)`. Schrijver's termination bound
gives at most `T=n(n+1)` dual adjustments. If input weights have at most `d`
digits, the split entries have at most
`D=d+ceil(log₁₀(2)+T log₁₀(3))` digits by the recurrence
`B'≤3B+2·10ᵈ`; at most `L=ceil(D/9)` decimal limbs are charged per arithmetic
scan. The scan bound is
`16 n²T + 4 n²(n+1) + 2 n²`. The kernel tests the admitted field prime once
and uses the admitted-prime rank entry point for exchange probes. Each `κᵢ`
charge dominates a selected-column matrix copy and canonical-residue checks.
Requests exceeding the combined 50,000,000-unit
rank-and-arithmetic envelope, the 1,024-digit intermediate cap, or the
retained-axis codepoint bound are rejected before rank computation. This favors
smaller grounds or low-row representations; the bound retains an exact finite
envelope rather than a wall-clock timeout.

The optimizer returns a replayable exchange-split witness, not a rank-dual
witness. The augmenting search's private Frank split only proves optimality
within each fixed cardinality and can fail the unrestricted split checker,
including on disjoint loop/nonloop sources; the returned split is synthesized
separately from final-candidate exchange inequalities and does satisfy that
checker. The separate rank-dual checker
accepts multiplier witnesses, but this kernel does not construct them. Treat this
optimizer as a prerequisite implementation, not completion of issue #1802's
weighted-intersection acceptance. A follow-up needs a bounded dual-producing
algorithm; the TDI and chain-support existence proof in [Goemans, Lecture
12](https://math.mit.edu/~goemans/18438F09/lec12.pdf) does not by itself
extract the multipliers.
