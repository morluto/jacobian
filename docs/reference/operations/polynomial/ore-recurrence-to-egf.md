# Polynomial recurrence to exponential generating function equation

`holonomic.shift_recurrence.to_egf_differential_equation.compute` maps a
polynomial recurrence relation

\[
\sum_{i=0}^{r}q_i(n)a_{n+i}=0\quad(n\ge0)
\]

to the exact homogeneous differential equation

\[
\sum_{i=0}^{r}q_i(xD)D^i E=0,
\qquad E(x)=\sum_{n\ge0}a_n\frac{x^n}{n!},
\]

where `D=d/dx`. In the exponential basis, `D^i E` has coefficient
`a_(n+i)` at `x^n/n!`, and the Euler operator `xD` acts on `x^n/n!` by
multiplication by `n`. Thus applying `q_i(xD)` gives exactly the exponential
generating function of the `i`th recurrence summand. Differentiation already
accounts for the shifted sequence entries, so this transform has no initial
term forcing.

The output is expanded into canonical left-coefficient form
`sum_j p_j(x)D^j`. For example, `n^2 a_n=0` maps through
`(xD)^2=xD+x^2D^2`. The operation requires coefficients in `QQ[n]`; rational
functions in `n` are rejected. Differential order is at most 16, coefficient
degree at most 64, and exact work and serialized output are bounded. It returns
the transformed relation; it does not assert that a sequence satisfies it or
that its EGF converges analytically.

The recurrence / D-finite correspondence is established in Richard P. Stanley,
[“Differentiably Finite Power Series”](https://doi.org/10.1016/S0195-6698(80)80051-5),
*European Journal of Combinatorics* 1 (1980), 175–188. The operator identity
above follows directly from the EGF basis and Euler-operator actions described
here.
