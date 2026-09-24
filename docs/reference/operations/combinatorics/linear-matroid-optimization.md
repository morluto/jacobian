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
both candidate feasibility ranks; the source-bound result also has a
conservative 8 MiB serialized-size bound that includes repeated axis labels.
A serialized result can be independently recomputed with
`verify_weighted_intersection_result`.
This operation checks a supplied split; it does not find an optimum candidate
or construct the split.

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
both candidate feasibility ranks, and the elementwise coverage scan. A
conservative 8 MiB result bound includes both sources, the dual terms, and the
derived split.

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
among the cardinality-optimal sets it computed. The split is private algorithm
state and is not returned as an unrestricted split certificate. Use one of the
separate certificate-checking operations when a caller supplies an authored
optimality witness. The algorithm and its correctness argument are in
[Schrijver and Korte–Vygen, §13.7](https://www.mathematik.uni-muenchen.de/~kpanagio/KombOpt/book.pdf).

The request admits at most 256 ground elements, the existing 11-decimal-digit
weight limit, and an 8 MiB source-bound result. Admission accounts for the
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
rank-and-arithmetic envelope, the 1,024-digit intermediate cap, or output
bound are rejected before rank computation. This favors smaller grounds or
low-row representations; the bound retains an exact finite envelope rather
than a wall-clock timeout.

This optimizer currently returns the exact candidate and objective without a
replayable rank-dual witness. Its internal Frank split only proves optimality
within each fixed cardinality and can fail the unrestricted split checker,
including on disjoint loop/nonloop sources. The separate rank-dual checker
accepts such witnesses, but this kernel does not construct them. Treat this
optimizer as a prerequisite implementation, not completion of issue #1802's
weighted-intersection acceptance. A follow-up needs a bounded dual-producing
algorithm; the TDI and chain-support existence proof in [Goemans, Lecture
12](https://math.mit.edu/~goemans/18438F09/lec12.pdf) does not by itself
extract the multipliers.
