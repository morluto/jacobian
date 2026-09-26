# Unnormalized chains of a finite simplicial set

`topology.simplicial_set.unnormalized_chain_complex.compute` turns a checked
table-based simplicial-set prefix `X_0,...,X_N` into the canonical based chain
complex over `ZZ`, `QQ`, or a bounded prime field `GF(p)`

```text
C_n = R[X_n]
d_n = sum(i=0..n) (-1)^i d_i
```

The result retains the source prefix and the ordered simplex labels for each
chain-group basis. Its `chain_complex` is the shared `ChainComplexValue`, so
existing chain-complex operations can consume it directly. Matrices use the
source's simplex order as columns and the preceding degree's order as rows.

The operation rechecks the supplied table identities because a caller may send
a serialized source value without trusted in-memory provenance. It admits
matrix cells and result size before allocating boundaries. The top group `C_N`
is included; the value stores differentials only through `d_N` and makes no
claim about `d_(N+1)`.

For example, the degree-0..2 prefix of `Delta[1]` has unnormalized ranks
`(2, 3, 4)`. The normalized complex has ranks `(2, 1, 0)` because it quotients
out degeneracies; use
[`topology.simplicial_set.normalized_chains.compute`](../../tools.md) for that
different chain complex.
