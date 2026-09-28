# First order lifting of a smooth local branch

`local_series.polynomial.smooth_branch_first_jet.compute` accepts a polynomial
in `y` whose coefficients are exact finite-center power-series prefixes in the
local parameter `t`, together with a rational value `c` satisfying
`F(0,c)=0` and `F_y(0,c) != 0`. It returns

```text
y(t) = c - F_t(0,c) / F_y(0,c) * t + O(t^2).
```

The coefficient series must be known through exponent one. This is enough to
determine the first jet: unknown coefficient tails begin at `t^2`, so they do
not affect the constant or linear residual. The operation checks the supplied
root and the nonzero derivative exactly before constructing the result.

This is the first-order consequence of the formal implicit function theorem:
every formal completion of the supplied coefficient prefixes has a unique
formal lift at a simple root, and all such completions have the same first
jet. The coefficient formula follows by comparing the coefficient of `t`
after substitution. See the [Stacks Project treatment of Hensel lifting in
complete local rings](https://stacks.math.columbia.edu/tag/04GE).

The operation supports smooth, unramified branches over `QQ` only. It does not
compute higher coefficients, choose an algebraic root, or lift multiple or
ramified branches. It is a direct first-jet result, not a claim about analytic
convergence or global branches.

Admission caps the source at 17 coefficient rows, `y` degree 16, 512 retained
series coefficients, 64-digit rational initial roots, and a 4096-digit bound
on every rational evaluation intermediate and output coefficient. The source
windows must provide at least precision two and contain no negative powers.

[Documentation home](../../../index.md) · [Polynomial operations](index.md) · [Tool surface](../../tools.md)
