# Disjoint Greene witnesses

[Documentation home](../../../index.md) · [Operation references](../index.md) · [Combinatorics](index.md)

`word.greene_witnesses.compute` returns one disjoint subsequence family for
each Greene index `1..k`. Positions are zero-based indices into the exact
source word. Under `ROW_INSERTION_RSK_V1`, increasing paths are weakly
increasing in the word's explicit alphabet order, and decreasing paths are
strictly decreasing.

For every requested index `j`, the returned family contains at most `j`
nonempty paths, no source position appears twice in that family, and the sum
of path lengths equals the corresponding RSK shape sum:

* the increasing total is `lambda_1 + ... + lambda_j`;
* the decreasing total is `lambda'_1 + ... + lambda'_j`.

The kernel computes maximum-cardinality disjoint path families directly with
unit-capacity min-cost flow on the finite position-order DAG. Entering a source
position contributes one unit of weight; transition edges encode the selected
monotonicity. The returned paths are one optimum. Ties can have multiple valid
solutions, and the operation makes no mathematical uniqueness claim about a
witness family.

The witness operation accepts words of length at most 32 and `k` from 1 to 8.
Before constructing the flow network, admission bounds the Bellman-Ford
relaxation estimate plus insertion and output bookkeeping by 6,000,000, and
aggregate witness memberships by 512.
Longer words can still use `word.greene_invariants.compute`, which returns the
shape-derived scalar totals without materializing witnesses.
