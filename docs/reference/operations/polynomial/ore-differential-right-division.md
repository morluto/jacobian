# Monic right division of differential Ore operators

`ore.differential.operator.right_division.compute` returns `Q` and `R` with

\[
A=QB+R,\qquad \operatorname{ord}(R)<\operatorname{ord}(B),
\]

in `QQ(x)<D>` with `D a=aD+a'`. The divisor is the right factor in `QB`;
this orientation is part of the operation contract. The result retains both
inputs and the typed quotient and remainder.

This slice admits operators of order at most four with integer polynomial
coefficients in `ZZ[x]`, and requires a nonzero divisor that is monic in `D`.
These conditions keep each cancellation step inside `ZZ[x]<D>` and make the
quotient and remainder unique. The operation preflights order, coefficient
degree, coefficient height, exact cancellation work, and output size before
expansion. The Euclidean division identity for differential Ore polynomials
is part of Ore's foundational treatment of noncommutative polynomials:
[Ore, “Theory of Non-Commutative Polynomials,” *Annals of Mathematics* 34
(1933), 480–508](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/ore.pdf).

For example, with `A=D^2+x` and `B=D+x`, right division gives
`Q=D-x`, `R=x^2+x-1`. The opposite factor order gives
`(D+x)D=D^2+xD`, which is not `A`; left and right division are distinct.

[Operation references](index.md) · [Tool surface](../tools.md)
