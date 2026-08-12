# Bounded Gaussian polynomial moments

[Documentation home](../../../index.md)

- Status: Experimental contract
- Capability: `probability.gaussian_polynomial.moment.compute`
- Verification: `probability.gaussian_polynomial.moment.verify`
- Producer: pinned Python-FLINT exact rational arithmetic
- Checker: isolated Python standard-library `Fraction` coefficient contraction

This capability computes one fixed-order moment

\[
  \mathbb E[P(X_1,\ldots,X_n)^m]
\]

where the \(X_j\) are independent standard real Gaussian variables and \(P\)
is a canonical sparse polynomial with coefficients in
\(\mathbb Q(i)\). It returns one exact complex-rational value and the complete
bounded coefficient-contraction ledger for that request. It is a mathematical
operation, not a Gaussian-identity search or proof workflow.

## Contract and scope

The request accepts either bounded loose wire terms or the strict canonical
`GaussianPolynomial` value emitted by parsing those terms. The bounds are:

| Field | Bound |
| --- | --- |
| Gaussian variables | 1–8 |
| Raw sparse terms | 1–16 before canonicalization |
| Canonical nonzero sparse terms | 1–16 after canonicalization |
| Total degree of each input term | at most 8 |
| Fixed moment order \(m\) | 0–16 |
| Raw ordered expansion paths | `term_count ** m <= 65536` |
| Each raw rational numerator/denominator | at most 128 decimal digits |
| Each strict canonical rational component | at most 4,096 decimal digits |

Loose wire terms use exponent vectors of length `variable_count`. They may arrive
in any order, repeat an exponent vector, or carry an exact zero coefficient. The
request boundary checks the raw 16-term and 128-digit limits before combining
duplicates over \(\mathbb Q(i)\), removes exact zero terms, and orders the
surviving terms lexicographically. Exact accumulation may grow beyond 128 digits,
but the strict canonical value retains the 4,096-digit component bound and can be
serialized and parsed again by the producer or checker without reapplying the
per-raw-term limit.

A request whose terms cancel completely is rejected, as are negative exponents,
inconsistent dimensions, more than 16 raw terms, canonical values with more than
16 nonzero terms, and requests beyond the complete-expansion bound. Direct
`GaussianPolynomial` values remain strict: terms must already be nonzero, unique,
and lexicographically ordered.

The producer exactly expands \(P^m\) via pinned Python-FLINT `fmpq_mpoly`
binary exponentiation over a typed real/imaginary coefficient pair,
merges equal exponent vectors, and removes zero coefficients. For every
remaining exponent vector
\(\alpha=(\alpha_1,\ldots,\alpha_n)\), its ledger records:

- the exact expanded coefficient;
- each univariate factor
  \(\mathbb E[X_j^{\alpha_j}]\), equal to zero for odd exponents and
  \((\alpha_j-1)!!\) for even exponents;
- the product of those factors; and
- the exact complex-rational contribution.

The result separately reports raw expansion-path count, canonical expanded
monomial count, `COMPLETE_BOUNDED_EXPANSION`, the Gaussian model
`INDEPENDENT_STANDARD_REAL`. It creates no verification record. Order zero
returns the moment one and its constant contraction.

## Claim boundary

A successful result claims only the exact value for the polynomial and single
integer order stored in its input artifact. It does not claim:

- vanishing or nonvanishing for any other order;
- an identity for all \(m\), eventual behavior, or an asymptotic;
- a mixed moment such as \(\mathbb E[Q P^m]\) unless that complete polynomial
  is itself supplied as the one bounded input;
- a result for correlated, nonstandard, or complex Gaussian variables; or
- support for algebraic coefficients outside \(\mathbb Q(i)\).

In particular, checking finitely many moments cannot establish the all-order
identities in
[Small Counterexamples to the Gaussian Moments Conjecture](https://arxiv.org/abs/2607.18186).
That public paper motivates the recurring contraction move, while an
all-parameter identity needs a different symbolic certificate and checker.

## Independent verification

The producer remains `COMPUTED`. The operator-authorized checker runs in an
isolated process, imports neither Python-FLINT nor the producer module, and
reconstructs the sparse power using only `Fraction`. It independently checks:

1. the exact source polynomial, dimension, order, and expansion bound;
2. every merged coefficient of the canonical expansion;
3. every odd/even univariate Gaussian factor;
4. every contribution and the final complex-rational sum;
5. the ledger length, ordering, completeness marker, and backend metadata; and
6. the artifact, semantics, candidate, scope, witness-format, and checker
   bindings supplied by the verification kernel.

Only an accepted replay of that exact stored result may return `VERIFIED`.
Missing entries, altered coefficients or factors, substituted sources,
malformed metadata, checker unavailability, timeout, and process failure
produce no verification record and no promoted conclusion.
