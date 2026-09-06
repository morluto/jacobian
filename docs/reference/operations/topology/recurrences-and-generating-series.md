# Recurrences and generating series

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Exact recurrence and generating-function operations belong to the
combinatorics domain:

- `combinatorics.generating_function.coefficients.compute`
- `combinatorics.recurrence.linear.evaluate`
- `combinatorics.recurrence.p_recursive.evaluate`
- `combinatorics.recurrence.p_recursive.table_residuals.compute`

They calculate finite typed prefixes and terms under the operation bounds. The
generating-function operation returns its prefix as the canonical
`TruncatedSeries` value (variable `x`, explicit truncation order, and exact
`QQ` coefficients) and retains the numerator and denominator presentation.
`verify_rational_generating_function_coefficients` checks the defining
congruence when a consumer relies on that source relation; decoding the value
checks only its structure. A finite prefix is not a claim about an infinite
series beyond the specified scope, and Jacobian does not retain a recurrence or
generated table.

`sequence.recurrence.closed_form.compute` returns a SymPy expression only for
characteristic polynomials of degree at most four. For exact bounded terms of a
higher-order constant-coefficient recurrence, use
`combinatorics.recurrence.linear.evaluate`.
