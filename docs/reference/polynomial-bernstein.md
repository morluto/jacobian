# Rational tensor-product Bernstein coordinates

`polynomial.bernstein.coefficients.compute` returns the unique coordinates of a
canonical `RationalPolynomial` on a nondegenerate `RationalBox` at a declared
multidegree. The native function is
`jacobian.math.polynomials.bernstein.bernstein_coefficients`.

For each ordered axis, write `x_i = a_i + (b_i-a_i)t_i`. The returned flat tensor
uses increasing lexicographic multi-indices, with the last axis varying fastest,
and the basis

\[
\prod_i {m_i\choose k_i}t_i^{k_i}(1-t_i)^{m_i-k_i}.
\]

`RationalBernsteinPolynomial` retains the canonical source polynomial, box,
multidegree, and exact rational coefficients. These values retain the complete
interpretation, including axis order; no additional digest or certificate is
needed. Deserialization checks structural interpretation without
reconstructing the polynomial. The native producer establishes the coefficient
identity once. Point boxes are rejected, and zero polynomials retain all axes
and the full declared tensor.

For `x²-x+y+1` on `[0,1]²` at degree `(2,1)`, the flat coefficients are
`(1, 2, 1/2, 3/2, 1, 2)`. Coefficient extrema give a convex-hull enclosure, not
necessarily the exact range. A nonpositive coefficient does not disprove
positivity. Subdivision and positivity-proof strategies remain caller-owned.
The existing `polynomial.box.enclosure.compute` returns a different value: an
interval enclosure, without these basis coordinates.

## Algorithm and backend

For each distinct source exponent `e` on an axis, compute one row of coordinates:

\[
T_{e,k}=\sum_{j=0}^{\min(e,k)}
{e\choose j}a^{e-j}(b-a)^j\frac{{k\choose j}}{{m\choose j}}.
\]

The tensor coefficient is the sparse sum of source coefficients times products
of these axis coordinates. This follows by affine substitution and the identity
`t^j = sum_{k=j}^m binom(k,j)/binom(m,j) B_k^m(t)`. It avoids both a dense global
basis-change matrix and an expanded multivariate affine polynomial.

The maintained [FLINT rational backend](https://python-flint.readthedocs.io/en/latest/fmpq.html)
owns exact rational arithmetic. Its
[dense univariate polynomial interface](https://python-flint.readthedocs.io/en/latest/fmpq_poly.html)
provides polynomial arithmetic but does not supply this source-bound tensor
representation. The adapter therefore owns the separable coordinate formula.
Inactive axes require no endpoint conversion or subtraction. FLINT's reduced
rational output is projected without repeating gcd computations. No child
process or symbolic expression parser is involved.

## Admission and execution envelope

Admission runs once in the native function, after canonical input validation.
Let `N = product(m_i+1)`, `S` be source support size, and `E_i` the distinct
source exponents on axis `i`. The implementation bounds:

- `N <= 65536`, including degree elevation of constants and zero polynomials;
- axis-map storage `sum_i |E_i|(m_i+1)` and tensor storage;
- sparse tensor accumulation and axis-map work, including binomial integer work,
  by `N*(S*(d+1)+1) + 8*sum_i (m_i+1)*sum_{e in E_i}(e+1)^2`;
- rational component growth before endpoint powers or binomial expansion.

A common denominator divides the product of source denominators, endpoint
lower/upper denominator products to coordinate degree, and
`product_{j=1}^{degree_i} binom(m_i,j)` on each axis. The estimate uses
`binom(m,j) <= m^j`, without expanding these integers during admission. Since
`binom(k,j)/binom(m,j) <= 1`, monomial coordinate magnitudes are bounded by
`(|a|+|b-a|)^e`. Combining these estimates, the source sum, and unreduced
binomial factors gives a component bound `H`. Kernel rational cross-products
fit `4H` decimal digits. Canonical input values separately own bounded scalar
normalization and interval ordering; the operation does not replay ordering.

Admission requires `H <= 8192` and height-weighted work
`work*(1+floor(H/64))² <= 20000000`. Tensor, map, and retained-source storage is
charged against four million estimated character units, with a 64-unit
per-coordinate allowance. This is an allocation estimate, not a transport byte
limit or a claim about measured resident memory. Transport retains its own
complete-envelope checks. Bounds are conservative: denominator products can
exceed their least common multiple. Raising this envelope requires a sharper
bound or algorithm and new accepted-case evidence.

The operation binds a 120-second cooperative safety deadline from the request's
original start, honoring an earlier caller deadline. Admission, axis maps,
tensor accumulation, source binding, and public finalization share that deadline;
native calls receive the same owner limit. Checkpoints also honor cancellation.
Deadline expiry is operational non-completion and returns no mathematical result.

## Evidence

Tests expand Bernstein basis polynomials directly in the original variables and
compare every monomial coefficient. They cover the issue's tensor, translated
rational boxes, degree elevation, constants, zero, transported axis permutations,
stale source bindings, point-box rejection, and inherited deadline expiry.
A separate repeated-degree-elevation oracle checks all 3,721 entries of a dense
bicubic transformed to degree `(60,60)` on a rational rectangle. Degree 4,096
for a linear polynomial is accepted without a dense global matrix.

A local calibration on 2026-09-06 used the dense bicubic with coefficient
`(-1)^(i+j)(i+1)(j+1)/7`, `0 <= i,j <= 3`, on 32 boxes
`[r/32,(r+1)/32] × [-1/3,2/3]`, at degree `(60,60)`. Parsing, native computation,
and JSON projection of 119,072 coefficients took 1.76 seconds in total; the
slowest request took 0.064 seconds. The last tensor had at most 16-digit scalar
components and a 171,080-byte serialized value. All corner identities were
checked independently. The 120-second owner deadline provides over 1,800 times
that measured per-request duration. These measurements calibrate a safety
margin; the admission proof supplies boundedness. This controlled family does
not replay or establish the motivating sine inequality.
