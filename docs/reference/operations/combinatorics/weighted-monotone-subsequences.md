# Weighted monotone subsequences

[Documentation home](../../../index.md) · [Operation references](../index.md) · [Combinatorics](index.md)

The `algebraic_combinatorics.weighted_monotone_subsequence.nondecreasing_maximum.compute`
and `...nonincreasing_maximum.compute` operations accept a finite ordered word
with one nonnegative exact rational weight for each position. They return the
maximum total weight of a weakly monotone subsequence and one deterministic
source-index witness. The two operation IDs name the direction explicitly;
there is no caller-selected strictness mode.

The dynamic program stores the best weight of a monotone subsequence ending at
each position. For position `i`, it adds its weight to the greatest prior state
whose letter is weakly below (or weakly above) the current letter. The kernel
examines every predecessor pair, so it establishes optimality in quadratic
time and retains predecessor indices for one witness. Equal-weight ties keep
the first improving predecessor and then the first endpoint. A nonempty source
returns a nonempty witness even when all weights are zero; an empty source
returns the empty witness of weight zero.

The exact rational inputs are reduced `CanonicalRational` values. Before the
quadratic scan, admission bounds each input component to 256 decimal digits,
the combined source rational payload to 4,096 digits, the conservative
numerator/denominator growth of any witness sum to 4,096 digits, and the
quadratic rational-arithmetic estimate to 20,000,000 units. With `n` source
positions and an admitted `d`-digit bound for every DP rational, the estimate
is `n(n-1)d² + 12nd²`: it charges two bigint-product units for each rational
comparison across possible predecessor pairs and a conservative bigint
allowance for each endpoint addition and the exact result scalar. Requests
outside that envelope are rejected before DP state construction. The source,
exact total, indices, and selected letters remain available together in the
result.

This finite sequence operation is related to the weighted monotone-subsequence
quantity discussed in Terence Tao's account of Erdős problem #1026, where the
endpoint states are denoted `S_i` and `T_i`. This operation computes one exact
finite optimum and witness; it makes no claim about the separate square-packing
inequality used in that proof. See [Tao's exposition](https://terrytao.wordpress.com/2025/12/08/the-story-of-erdos-problem-126/)
and Baek, Koizumi, and Ueoro's [square-packing paper](https://arxiv.org/abs/2411.07274).
