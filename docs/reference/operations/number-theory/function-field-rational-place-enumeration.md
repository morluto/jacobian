# Rational function-field place enumeration

`function_field.places.degree_bounded.enumerate` returns the complete set of
places of `GF(p)(x)` of degree at most `d`. The result includes the unique
infinite place of degree one and every finite place represented by its unique
monic irreducible polynomial in `GF(p)[x]`. Finite place degree is polynomial
degree.

The operation currently accepts only the rational function field over a prime
field. Place enumeration for a nontrivial finite extension requires factoring
places in a model-specific integral closure and is outside this operation's
contract.

For each degree `n` the kernel enumerates all `p^n` monic polynomials and uses
an exact finite-field irreducibility test. The candidate count
`sum(p^n, n=1..d)`, a degree-weighted irreducibility work estimate, and the
maximum possible output size are checked before enumeration. Requests that
exceed any envelope fail with a typed resource admission error; no partial list
is returned as complete. Each finite output is an irreducible monic polynomial,
and exhaustive traversal of all monic candidates establishes completeness.

The finite-place representation agrees with the usual correspondence between
closed points of the affine line over a finite field and monic irreducible
polynomials; Sage's finite-field polynomial reference documents exact
irreducibility operations [for polynomials over finite fields](https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polynomial_element.html).

The request admits degree at most 12, at most 16,384 polynomial candidates,
20,000,000 estimated irreducibility work units, and at most 16,385 output
places. `d=1` includes all `p` rational finite places and infinity.

The operation does not enumerate places on algebraic extensions or return
residue fields, uniformizers, or maps between places. It is a complete bounded
enumeration for the rational field only.
