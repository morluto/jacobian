# Free associative algebra polynomial powers

[Documentation home](../../index.md) · [Tool surface](../tools.md) · [Operation references](index.md)

`free_algebra.polynomial.power.compute` returns the exact nonnegative integer
power of one sparse `QQ`-linear polynomial in the free associative algebra. The
result is a canonical `FreeAlgebraPolynomial` over the input alphabet. At
exponent zero, the result is the unit polynomial over that same alphabet;
exponent one preserves the input value, and every positive power of zero is
zero.

The exponent is limited to 64. Larger powers use exponentiation by squaring and
the existing polynomial multiplication kernel. Before each intermediate
convolution, admission checks the term-pair count, the result support, and the
predicted coefficient growth. Each accepted convolution is exact and complete.
As with polynomial multiplication, operand supports are limited to 64 terms and
operand words to 32 letters; a result may contain up to 4,096 terms and
64-letter words.

Those bounds count retained terms and predicted coefficient width, not
retained letters. There is no separate output-allocation bound on the total
word content of a result, so the letter content of an admitted result is a
function of the term and word-length limits above rather than a separately
enforced ceiling.

Because multiplication is noncommutative, powers preserve word order. For
example `(x+y)^2 = xx + xy + yx + yy`, with `xy` and `yx` remaining distinct.
