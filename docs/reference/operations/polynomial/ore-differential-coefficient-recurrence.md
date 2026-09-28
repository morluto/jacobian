# Differential equation to Taylor coefficient recurrence

`holonomic.differential_operator.to_coefficient_recurrence.compute` converts
an operator

\[
L=\sum_j p_j(x)D^j,\qquad p_j(x)\in\mathbb Q[x],
\]

into the exact coefficient equations for a formal series
\(f(x)=\sum_{k\ge0}a_kx^k\). For a monomial \(c x^lD^j\), the contribution
to the coefficient of \(x^m\) is

\[
c\,(m-l+j)_{\underline j}\,a_{m-l+j}
\]

when \(m\ge l\), and zero otherwise. Rows before the largest polynomial
coefficient degree are returned explicitly as `boundary_rows`. Once every
coefficient monomial is active, the stable recurrence is returned as
\(\sum_i q_i(n)a_{n+i}=0\), with its first valid index `valid_from`.
The recurrence uses the canonical `ShiftOreOperator` value, so it can be
serialized directly into `ore.shift.recurrence.generate_finite_prefix.compute`;
pass `valid_from` as that operation's `start_index` and supply the needed
initial values. The shared shift-operator value admits exponents through 80
for this recurrence domain; arithmetic and power operations retain their own
tighter work and output limits.

The operation accepts polynomial coefficients over \(\mathbb Q[x]\) only.
It preserves rational arithmetic and does not choose initial values, claim a
unique solution, or assert convergence. It admits the input expansion work,
coefficient height, recurrence shift span, and output size before constructing
the recurrence. The exact coefficient comparison is the standard formal-series
method described for ordinary points in [NIST DLMF §2.7](https://dlmf.nist.gov/2.7).

[Polynomial operations](index.md) · [Operation catalog](../../tools.md)
