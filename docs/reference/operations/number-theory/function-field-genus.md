# Rational function-field genus

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.genus.compute` accepts only the rational function field
`GF(p)(x)`, represented by `FiniteFunctionField` with a prime characteristic
and defining-polynomial tuple `(1,)`. The model carries a `variable` name; its
`generator` label is unused in this rational-field branch. A nontrivial
algebraic extension, even if otherwise valid, is outside this operation's
domain and is rejected with `function_field.genus_requires_rational_field`.

The operation returns `0` because the unique smooth projective curve associated
to `GF(p)(x)` is the projective line `P^1` over `GF(p)`, whose genus is zero.
Using the two standard affine charts, its first structure-sheaf cohomology is
the Laurent-polynomial quotient
`GF(p)[x,x^-1] / (GF(p)[x] + GF(p)[x^-1]) = 0`: every Laurent polynomial is
the sum of its nonnegative and nonpositive terms. Since genus is
`dim H^1(P^1, O)`, this gives the exact value. The Stacks Project defines genus
for smooth projective curves as this cohomology dimension in its
[curve-genus reference](https://stacks.math.columbia.edu/tag/0BY6).
Admission revalidates the field carrier and primality of `p` before recognizing
the rational-field representation; the computation performs no extension
factorization or place enumeration. The wire carrier bounds `p` by 257, so
primality admission is also bounded.

This operation does not compute genus for algebraic extensions. That requires
the missing exact extension place model and a complete differential or
Riemann–Roch method. It also does not establish a canonical divisor or any
Riemann–Roch space.
