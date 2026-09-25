# Finite prefixes of smooth local branches

`local_series.polynomial.smooth_branch_prefix.compute` computes the unique
formal series `y(t) = c + c_1 t + ... + c_(N-1) t^(N-1) + O(t^N)` satisfying
`F(t, y(t)) = O(t^N)`, when the input supplies a rational simple root `c` of
`F(0,y)`. The polynomial coefficients are exact rational power-series
prefixes at one finite center, each known through exponent `N-1`.

At each order, the operation substitutes the already determined lower-order
terms into `F`. The coefficient of the new term is multiplied by the nonzero
constant `F_y(0,c)`, so exactly one rational coefficient cancels the next
residual term. This is finite formal Hensel lifting, the higher-order
counterpart of the [first-jet operation](local-series-smooth-branch-first-jet.md).
The formal implicit function theorem gives existence and uniqueness over
`QQ[[t]]`; see the [Stacks Project treatment of Hensel lifting in complete local
rings](https://stacks.math.columbia.edu/tag/04GE).

The output is bound to the source polynomial, root, local parameter, center,
and exclusive precision. It proves a congruence modulo `t^N`; it makes no
claim about analytic convergence, global branches, algebraic initial roots,
multiple roots, or ramification.

Admission limits the precision to 3 through 32, the `y` degree to 16, the
source to 17 coefficient rows and 512 retained rational slots, input
coefficients to 256 digits, the initial root to 64 digits, and every predicted
Hensel coefficient/intermediate to 4096 digits. A conservative rational-height
recurrence and the full Horner work bound are checked before branch expansion.

[Documentation home](../../../index.md) · [Polynomial operations](index.md) · [Tool surface](../../tools.md)
