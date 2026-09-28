# Shift Ore operator powers

`ore.shift.operator.power.compute` returns the exact nonnegative power
`P^m` of a shift operator in left-coefficient normal form

\[
P=\sum_i p_i(n)S^i,
\qquad S^i a(n)=a(n+i)S^i.
\]

The exponent is at most 16. Exponent zero returns the multiplicative
identity. The operation rejects any input whose power would exceed the
operation's result-order limit of 16. At most 16 products of at most 16-by-16
terms are expanded; each product also uses the shift multiplication
operation's exact coefficient-degree, digit, and output-carrier admissions.
For powers above one, coefficients must lie in `ZZ[n]`; this lets admission
bound every stage's degree, support, and integer coefficient height before
any product is expanded. The result is an ordinary `ShiftOreOperator` and can
be passed to finite sequence-prefix action.

This is a current admission limit, not a mathematical restriction on powers in
`QQ(n)<S>`. For example, `(1/n+S)^2` is finite and exact, but its intermediate
rational-function normalization growth is not yet bounded by the whole-power
admission estimate, so exponents above one with rational-function coefficients
are rejected before multiplication. Exponents zero and one still accept the
full rational-function coefficient carrier. Extending the higher-power
envelope requires a sound stage bound for shifted rational functions and their
reduced sums/products.

For constant coefficients, `(1+S)^3 = 1+3S+3S^2+S^3`. The regression checks
this independently with a rational binomial-coefficient oracle and confirms
its action on a geometric sequence prefix. The variable-coefficient identity
`(n+S)^2=n^2+(2n+1)S+S^2` exercises the shift commutation in a power.

[Shift-operator addition and scaling](ore-shift-polynomial-algebra.md) ·
[Finite-prefix action](ore-shift-sequence-prefix.md) ·
[Operation catalog](../../tools.md)
