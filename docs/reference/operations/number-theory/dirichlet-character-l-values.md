# Exact Dirichlet-character L-values at nonpositive integers

`dirichlet_character.l_value_nonpositive_integer.compute` returns the exact
algebraic value

\[
L(1-k,\chi)=-\frac{B_{k,\chi}}{k}
\]

for `1 <= k <= 32`, using the same generalized Bernoulli convention as
[`dirichlet_character.generalized_bernoulli.compute`](dirichlet-character-generalized-bernoulli.md).
The result retains the source character, the Bernoulli index `k`, the exact
argument `1-k`, and a canonical rational cyclotomic value in the character's
value field. This is a finite exact special-value computation; it does not
evaluate the analytic L-function at a general complex argument.

The existing Bernoulli work, coefficient-growth, and result-byte bounds apply.
The additional denominator factor `k` is included in the coefficient admission
before the character sum is expanded.

For the nonprincipal character modulo 3, `B(1,chi)=-1/3`, so `L(0,chi)=1/3`.
The trivial character modulo 1 at `k=1` gives `L(0)=-1/2` under the
`B_1(x)=x-1/2` polynomial convention.
