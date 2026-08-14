# Rational-function identities

`polynomial.rational_function.identity.verify` decides equality of two bounded
sparse rational functions in a declared fraction field
`QQ(x1, ..., xn)`.

The request preserves each numerator and denominator as a canonical sparse
polynomial. Denominators must be nonzero polynomials. An authorized independent
checker compares

```text
left.numerator * right.denominator
    = right.numerator * left.denominator
```

using exact rational arithmetic. The resulting certificate and verification
record are bound to both rational-function artifacts, the identity claim, the
semantics descriptor, and the checker identity.

This is fraction-field equality. It does not claim that either expression is
defined at a particular point, that their pointwise domains agree, or that a
larger theorem containing the identity is true. Those are separate
obligations.

The bounded contract accepts one to four ordered variables, at most 1024 terms
per component polynomial, exponents at most 127, and at most 4096 term pairs in
either cross product.
