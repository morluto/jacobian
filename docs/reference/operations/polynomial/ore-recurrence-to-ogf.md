# Polynomial recurrence to ordinary generating function equation

`holonomic.shift_recurrence.to_ogf_differential_equation.compute` maps a
polynomial recurrence relation

\[
\sum_{i=0}^{r}q_i(n)a_{n+i}=0\quad(n\ge0)
\]

to an exact differential equation `L(F)=B` for the ordinary generating
function `F(x)=sum_(n>=0) a_n*x^n`. It takes the first `r` coefficients as
boundary data and returns

\[
L=\sum_i x^{r-i}q_i(xD-i),\qquad
B=\sum_i x^{r-i}q_i(xD-i)\sum_{m=0}^{i-1}a_mx^m,
\]

where `D=d/dx`. The powers of the Euler operator are expanded into the
canonical left-coefficient form `sum_j p_j(x)D^j`. The equation follows by
writing `sum_(n>=0) a_(n+i)x^n = x^(-i)(F-sum_(m<i)a_mx^m)` and multiplying by
`x^r` to clear all negative powers. This is an identity between the recurrence
relation and its generating function; it does not verify an infinite sequence,
assert that a particular `F` satisfies the recurrence, or claim convergence.

The operation requires coefficients in `QQ[n]`, exactly `r` initial values,
differential order at most 16, cleared coefficient degree at most 64, and
bounded exact work and output. Rational-function recurrence coefficients are
not accepted. This polynomial recurrence / D-finite correspondence and its
operator forms are treated in [Kauers, *D-Finite Functions*, chapters 2–3](https://link.springer.com/book/10.1007/978-3-031-34652-1).

[Polynomial operations](index.md) · [Operation catalog](../../tools.md)
