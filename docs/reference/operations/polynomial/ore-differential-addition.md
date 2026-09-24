# Differential Ore operator addition

`ore.differential.operator.add.compute` adds operators in the left-coefficient
normal form

\[
P=\sum_i a_i(x)D^i,\qquad a_i(x)\in\mathbb Q(x),\quad
D a(x)=a(x)D+a'(x).
\]

The inputs must use the same structural `x` axis. Addition is coefficientwise:
coefficients at equal derivative orders are added as exact rational functions,
orders are sorted, and exact zero coefficients are removed. The empty term tuple
is the canonical zero operator. This operation does not multiply the operators,
clear coefficient denominators, or claim any scalar normalization.

Each input has at most 32 terms, derivative order at most 16, and coefficient
representations bounded to 64 terms, degree, and coefficient digits. Before
rational-function addition, the kernel admits the coefficient degrees,
denominator degrees, coefficient heights, sparse convolution work, polynomial
normalization estimate, and serialized result size. Results must fit the
canonical rational-function carrier (degree 128, 256 terms, and 128 coefficient
digits) and the operation's work and output budgets.

The operation composes with differential-operator action and multiplication:
applying a sum to a rational function agrees with the sum of the two actions.
The general Ore relation and twisted multiplication convention are described
in the [SageMath Ore-polynomial reference](https://doc.sagemath.org/html/en/reference/noncommutative_polynomial_rings/sage/rings/polynomial/ore_polynomial_element.html).

[Polynomial operations](index.md) · [Operation catalog](../../tools.md)
