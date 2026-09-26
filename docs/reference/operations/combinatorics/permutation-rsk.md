# Permutation RSK and its inverse

[Documentation home](../../../index.md) · [Operation references](../index.md) · [Combinatorics](index.md)

`FinitePermutation` stores the one-line images of a bijection of `1..n`; the
empty tuple is the permutation of the empty set. The operation
`combinatorics.rsk.permutation.compute` returns a source-free
`PermutationRSKPair` containing standard insertion and recording tableaux of
the same shape. The shape is derived from the tableaux. The pair does not echo
the input permutation.

Both directions use `ROW_INSERTION_RSK_V1`. Forward insertion processes the
permutation from left to right and bumps the first entry strictly greater than
the inserted value. `tableau.rsk.inverse_permutation.compute` removes recording
labels from `n` down to `1`; each label locates an outer corner of the insertion
tableau, and reverse row insertion bumps the rightmost strictly smaller entry
through the rows above it. The emitted entries in reverse order form the unique
permutation of `1..n` represented by the pair. This is the standard bijection
between permutations and pairs of standard Young tableaux of a common shape;
see Knuth, [“Permutations, matrices, and generalized Young tableaux”](https://doi.org/10.2140/pjm.1970.34.709).

The permutation and pair carriers are bounded to 500 entries/cells. Each
direction visits at most `n(n-1)/2` row positions, with at most
`ceil(log2(n+1))` comparisons per binary row search. The tableau pair contains
exactly `2n` cells. Inverse admission validates the canonical same-shape pair
and bounds this work before reverse insertion. Its result is a validated
`FinitePermutation`, so it can feed the forward request directly after normal
request wrapping and serialization.

The specialized permutation pair is distinct from ordinary word RSK: its two
tableaux are both standard and its entries are integers `1..n`. The word-RSK
pair retains a string alphabet and has a semistandard insertion tableau, so the
two values are not coerced into one another.
