# Formal q-series U and V prefix maps

These operations apply exact finite coefficient maps to one `TruncatedSeries`
in the variable `q`:

\[
U_p\left(\sum_n a_n q^n\right)=\sum_n a_{pn}q^n,
\qquad
V_p\left(\sum_n a_n q^n\right)=\sum_n a_nq^{pn}.
\]

The result is another `TruncatedSeries` at the explicitly requested output
precision. These are formal power-series maps only. They do not establish that
the input or output is a modular form, and they do not return a modular-space
parent. Use the coordinate-based modular-form operators when a modularity
postcondition is required.

For output precision `P` and prime `p`, `U_p` reads through source order
`p(P-1)+1`. `V_p` reads through source order
`floor((P-1)/p)+1`; output coefficients at exponents not divisible by `p` are
exact zero. A request is rejected if this required source order exceeds the
represented prefix. Neither map treats unknown coefficients beyond the prefix
as zero.

The series carrier starts at order one, so the zero series is represented by
the one-term prefix `(0)` rather than an empty coefficient tuple.

The operations admit prime `p` through 10,000, source order through 25,280,
output precision through 4,096, selected coefficient numerator/denominator
digits through 4,096, a bounded linear work budget, and the canonical 10 MiB
aggregate JSON output limit before constructing the result. Because they only
select existing rational coefficients or insert zero, coefficient height does
not grow.

[Number-theory operations](index.md) · [Tool surface](../../tools.md)
