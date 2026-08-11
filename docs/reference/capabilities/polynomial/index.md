# Polynomial capability references

[Documentation home](../../../index.md) · [Capability surface](../../tools.md)

- [Bounded Gaussian polynomial moments](gaussian-polynomial-moments.md)
- [Typed polynomial expression normalization](polynomial-expression-normalization.md)
- [Polynomial-map Keller and inverse-obstruction verification](polynomial-map-conjecture-verification.md)
- [Polynomial-map inverse synthesis and verification](polynomial-map-inverse-verification.md)
- [Rational-function identities](rational-function-identities.md)

## Exact sparse polynomial values

`RationalPolynomial` is the shared inline value for exact polynomials over
`QQ`. It preserves the declared variable order, descending lexicographic sparse
term order, omitted zero terms, and canonical reduced rational coefficients.
`RationalPolynomialMap` is deliberately rectangular: square-map requirements
belong to operations such as Jacobian determinants and inverse verification.

The shared representation accepts up to eight variables or coordinates, 4,096
terms, exponents through 32,768, and canonical rational components through
32,768 digits. These are representation envelopes, not execution promises.
Each operation preflights its own degree, coefficient, expansion, enumeration,
and output limits before converting to SymPy. A structurally valid value can
therefore be rejected as too expensive for a particular operation.

SymPy conversion is owned by the polynomial domain. Capability operations use
the same typed producer kernels as the small native
`jacobian.math.polynomials` API; neither route serializes an intermediate
polynomial through JSON. Independent checkers consume the authoritative
contracts and implement their own relation checks rather than importing these
producer conversions or kernels.

## Independent verification

Exact polynomial producers and consumers exchange `RationalPolynomial` and
`RationalPolynomialMap` values directly. The corresponding replay checkers
receive the validated inline request and candidate, independently recompute the
declared relation, and bind the canonical input, candidate, semantics, witness
format, and checker identity. They do not call polynomial producer kernels or
reuse their conversions.

An accepted replay persists a verification record plus the immutable semantics
artifact it binds; those two URIs are exposed in `artifact_uris`. The ordinary
polynomial input and candidate remain inline. Rejection, timeout, cancellation,
malformed values, and unsupported relations create no record and remain
non-conclusions.
