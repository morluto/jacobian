# Free associative algebra polynomial subtraction

`free_algebra.polynomial.subtract.compute` computes the exact difference of
two sparse polynomials over `QQ`. Both operands must use the identical ordered
generator alphabet. Coefficients on equal words are subtracted; zero
coefficients are removed; distinct words such as `(x,y)` and `(y,x)` remain
distinct. The result uses the existing canonical `FreeAlgebraPolynomial`
value, including the empty-support zero polynomial and its alphabet.

The operation checks canonical input structure and alphabet identity, then
admits operand terms, coefficient growth, exact work, and result allocation
before coefficient aggregation. Each operand may have at most 128 terms and
the combined support bound is 128 terms. The exact predicted output allocation
may not exceed 150,000 cells.

For example,

```text
(2x + 3/2 xy) - (x + y) = x + 3/2 xy - y.
```

The operation is exact sparse arithmetic only; it does not reduce modulo an
ideal or identify words through relations.
