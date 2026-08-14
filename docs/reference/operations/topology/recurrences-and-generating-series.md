# Recurrences and generating series

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Exact recurrence and generating-function operations belong to the
combinatorics domain:

- `combinatorics.generating_function.coefficients.compute`
- `combinatorics.recurrence.linear.evaluate`
- `combinatorics.recurrence.p_recursive.evaluate`
- `combinatorics.recurrence.p_recursive.table_residuals.compute`

They calculate finite typed prefixes, terms, and residuals under the operation
bounds. A finite prefix is not a claim about an infinite series beyond the
specified scope, and Jacobian does not retain the recurrence or generated
table.
