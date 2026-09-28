# Free associative algebra polynomial powers

`free_algebra.polynomial.power.compute` returns the exact nonnegative integer
power of one sparse `QQ`-linear polynomial in the free associative algebra. The
result is a canonical `FreeAlgebraPolynomial` over the input alphabet. At
exponent zero, the result is the unit polynomial over that same alphabet;
exponent one preserves the input value, and every positive power of zero is
zero.

The exponent is limited to 64. Larger powers use exponentiation by squaring and
the existing polynomial multiplication kernel. Before each intermediate
convolution, admission checks its term-pair count, result support, coefficient
growth, and a 1,500,000-cell output allocation estimate. Each accepted
convolution is exact and complete. As with polynomial multiplication, operand
supports are limited to 64 terms and operand words to 32 letters; a result may
contain up to 4,096 terms and 64-letter words.

Because multiplication is noncommutative, powers preserve word order. For
example `(x+y)^2 = xx + xy + yx + yy`, with `xy` and `yx` remaining distinct.
