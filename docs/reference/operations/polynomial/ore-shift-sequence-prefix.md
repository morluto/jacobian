# Shift Ore operator on a finite sequence prefix

`ore.shift.operator.apply_to_sequence_prefix.compute` evaluates

\[
P f(n)=\sum_i p_i(n)f(n+i)
\]

for a shift operator in left-coefficient form over `QQ(n)` and a finite
`FiniteRationalSequence`. The request supplies `start_index`, which assigns
the first stored value to that integer; the sequence value itself has no
implicit index origin.

The result returns each exact coefficient, source value and contribution,
followed by the residual at that index. An index where any coefficient has a
pole is listed in `coefficient_poles` with the affected shift exponents and
has no residual row. Indices at the right edge that need values beyond the
supplied prefix are listed in `right_boundary_indices`.

This operation checks a finite prefix only. A zero residual on these rows is
not a global annihilation claim and does not construct a `PRecursiveSequence`.
The index range, coefficient-evaluation work, exact rational growth, and
serialized output are admitted before coefficient evaluation. Rational
coefficient poles remain exclusions; they are never treated as zero-valued
coefficients.

[Operation catalog](../../tools.md) · [Polynomial operations](index.md)
