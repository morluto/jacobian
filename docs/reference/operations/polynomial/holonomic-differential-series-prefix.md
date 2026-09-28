# D-finite Taylor prefix

`holonomic.differential_series.generate_finite_prefix.compute` computes the
first `count` Taylor coefficients at the ordinary point (x=0) from a
`DFinitePowerSeries` value. The output is the ordinary
`FiniteRationalSequence` value, with entry (a_i) equal to the coefficient of
(x^i). The input stores derivatives (f^{(i)}(0)), so its first `r` output
coefficients are (f^{(i)}(0)/i!), where `r` is the differential-operator
order.

For an operator
\[
L=\sum_{j=0}^r p_j(x)D^j,\qquad
p_j(x)=\sum_k c_{j,k}x^k,
\]
coefficient comparison at degree \(m\) gives
\[
\sum_{j=0}^{r}\sum_{k=0}^{m}c_{j,k}\frac{(m-k+j)!}{(m-k)!}a_{m-k+j}=0.
\]
The coefficient of \(a_{m+r}\) is
\(c_{r,0}(m+1)(m+2)\cdots(m+r)\), which is nonzero at every
nonnegative integer \(m\) because the input is an ordinary-point value.
Therefore each next coefficient is uniquely determined. The same equation
for order zero gives the zero series.

The current arithmetic domain requires each \(p_j\) to lie in
\(\mathbb{Q}[x]\), even though the input carrier also represents regular
rational-function coefficients. Rational coefficients need a separately
bounded denominator-series inversion and are not accepted by this operation.
The underlying ordinary-point coefficient-comparison principle is stated in
[NIST DLMF §2.7](https://dlmf.nist.gov/2.7).

Admission precedes recurrence arithmetic. The implementation estimates a
common integer scale for coefficient denominators and coefficient magnitudes,
then iterates conservative
numerator and shared-denominator bit bounds derived from the displayed
recurrence. The resulting accounting bounds coefficient multiplication,
fraction addition and pivot division. It also checks per-rational digits,
aggregate sequence digits, the 100,000-entry sequence limit, and serialized
output bytes. The work budget is 100,000,000 scalar-bit units; the coefficient
scale is limited to 131,072 bits; output is limited to 12 MiB. Empty prefixes
are valid. This result contains only the requested finite values: it does not
assert convergence or expose an infinite sequence.
