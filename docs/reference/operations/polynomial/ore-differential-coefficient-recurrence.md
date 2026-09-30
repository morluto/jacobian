# Differential equation to Taylor coefficient recurrence

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

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

when \(m\ge l\), and zero otherwise. The normalized recurrence coordinate is
\(n=m+\min(j-l)\), and the stable recurrence is
\(\sum_i q_i(n)a_{n+i}=0\). Its `valid_from` is at least the index where every
coefficient monomial is active and lies after every integral root of the
highest-shift coefficient. All earlier Taylor-degree equations, including
those passed while avoiding singular indices, are retained in `boundary_rows`.
Their exclusive cutoff is `valid_from - min(j-l)`, expressed in the original
Taylor coordinate rather than the normalized recurrence coordinate.
The recurrence uses the canonical `ShiftOreOperator` value, so positive-order
outputs can be serialized directly into
`ore.shift.recurrence.generate_finite_prefix.compute`; pass `valid_from` as
that operation's `start_index` and supply the needed initial values. An
order-zero output (for example, an Euler-type equation) is still a valid
coefficient equation but is not a finite-prefix recurrence and cannot be
handed to that operation. The shared shift-operator value admits exponents
through 16, its `MAX_SHIFT_ORDER` carrier limit, and a generated recurrence with
a wider shift span is refused rather than truncated. Boundary rows index Taylor
coefficients, not shifts, so their indices run to the row-degree cap plus that
differential-order limit. At most 64 consecutive Taylor boundary rows are
materialized; a later integral singularity requiring more rows produces a typed
resource refusal before expansion. The normalized start can exceed 16 and is
admitted separately from the recurrence shift order. Additional boundary work
is charged against the shared conversion budget. Arithmetic and power
operations retain their own tighter work and output limits.

The operation accepts polynomial coefficients over \(\mathbb Q[x]\) only.
It preserves rational arithmetic and does not choose initial values, claim a
unique solution, or assert convergence. It admits the input expansion work,
coefficient height, recurrence shift span, and output size before constructing
the recurrence. The exact coefficient comparison is the standard formal-series
method described for ordinary points in [NIST DLMF §2.7](https://dlmf.nist.gov/2.7).

[Polynomial operations](index.md) · [Operation catalog](../../tools.md)
