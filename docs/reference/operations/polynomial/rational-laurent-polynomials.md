# Exact rational Laurent-polynomial multiplication

[Documentation home](../../../index.md) · [Polynomial operations](index.md) ·
[Tool surface](../../tools.md)

`polynomial.laurent.rational.multiply.compute` multiplies two finite sparse
Laurent polynomials over the explicitly declared ordered `QQ` variable axis.
Each term has a signed integer exponent tuple and a nonzero canonical rational
coefficient. Equal exponent tuples are collected exactly; terms whose reduced
coefficient is zero are omitted. An empty support is the zero value and still
retains its variable axis, so it can be passed directly to a later operation.

The defining identity is

```text
(Σ_a c_a x^a)(Σ_b d_b x^b) = Σ_e (Σ_{a+b=e} c_a d_b) x^e.
```

The result is deterministic: support is unique and descending
lexicographic, coefficients are reduced rationals, and the ordered axis is
copied unchanged. Factors with different axes are a domain error. Negative,
positive, mixed, and zero exponent vectors are all ordinary finite Laurent
support; this operation has no convergence, valuation, or truncation meaning.

## Semantic admission

The shared representation admits at most eight variables, 4,096 nonzero
terms, and absolute exponents of 32,768. The related PR #3653 supplies the
signed minimum and maximum pair-sum preflight on every axis; merge that
contract repair before this follow-up. This change adds the sparse pair-work
bound (at most 4,096 products), a conservative exact rational-height bound,
and monomial shift/scale presolve. These checks are based on mathematical
support, coefficient components, and derived work—not on transport bytes or a
dense exponent box. Zero inputs return immediately while preserving the left
parent.

After the one exact convolution, reduced output coefficients are checked
against the canonical rational limit before result construction. Cancellation
therefore reduces the returned support rather than being mistaken for a dense
box requirement. Admission and kernel checkpoints share the request deadline;
result parsing does not replay the convolution. The public operation ID and
value schema are unchanged, so #3653 and this contract follow-up compose
without a new or superseding declaration.

Python-FLINT 0.9.0 provides sparse `fmpq_mpoly` arithmetic, but its
nonnegative monomial representation does not carry signed Laurent exponents.
The owner therefore uses bounded Python exact-rational convolution for this
small admitted support, keeping the Laurent parent and exponent semantics
directly in Jacobian's canonical value. FLINT remains suitable for a future
backend when a larger Laurent representation with an explicit shift is added.

## Native API

```python
from jacobian.math import polynomials

product = polynomials.rational_laurent_multiply(left, right)
```

The native function and the catalog operation consume and return the same
`RationalLaurentPolynomial` value. Infinite Laurent or Puiseux series remain a
separate series/truncation domain.
