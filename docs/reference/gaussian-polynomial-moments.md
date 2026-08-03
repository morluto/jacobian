# Bounded Gaussian polynomial moments

[Documentation home](../index.md)

- Status: Current implementation reference; contract is experimental
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

The request bounds are:

| Field | Bound |
| --- | --- |
| Gaussian variables | 1–8 |
| Nonzero sparse terms | 1–16 |
| Total degree of each input term | at most 8 |
| Fixed moment order \(m\) | 0–16 |
| Raw ordered expansion paths | `term_count ** m <= 4096` |
| Input rational numerator/denominator | at most 128 decimal digits |

Terms use exponent vectors of length `variable_count`, in strictly increasing
lexicographic order. A coefficient has separate canonical rational `real` and
`imaginary` components. Duplicate monomials, zero coefficients, negative
exponents, inconsistent dimensions, and requests beyond the complete-expansion
bound fail validation before computation or artifact writes.

The producer exactly expands \(P^m\), merges equal exponent vectors, and removes
zero coefficients. For every remaining exponent vector
\(\alpha=(\alpha_1,\ldots,\alpha_n)\), its ledger records:

- the exact expanded coefficient;
- each univariate factor
  \(\mathbb E[X_j^{\alpha_j}]\), equal to zero for odd exponents and
  \((\alpha_j-1)!!\) for even exponents;
- the product of those factors; and
- the exact complex-rational contribution.

The result separately reports raw expansion-path count, canonical expanded
monomial count, `COMPLETE_BOUNDED_EXPANSION`, the Gaussian model
`INDEPENDENT_STANDARD_REAL`, and `UNVERIFIED`. Order zero returns the moment
one and its constant contraction.

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

## Development handoff

### Discovery

- Status: accepted for the bounded fixed-order outcome only.
- Evidence: arXiv:2607.18186 exposes repeated exact Gaussian coefficient
  contractions; the pre-change catalog had 282 capabilities and only
  finite-distribution probability moments.
- Portfolio delta: this operation integrates a polynomial against the standard
  Gaussian product measure. It does not duplicate
  `probability.finite_distribution.raw_moment.compute`, which sums an explicit
  finite atomic distribution, or generic polynomial normalization, which has
  no probability semantics.
- Public reproduction:
  [`gaussian-sixth-moment`](../../benchmarks/datasets/public-reproductions-v1/gaussian-sixth-moment/).
- Evaluation hypothesis: a domain-owned fixed-order outcome plus independent
  replay should reduce hand-written Wick arithmetic errors without increasing
  false all-order claims or false `VERIFIED` results.

### Implementation

- Base revision: `7c205f59bbeb0ea6bb095fb1aaf22be28094d2ab`.
- Runtime: CPython 3.12; pinned `python-flint==0.9.0`; no dependency change.
- Live post-change catalog: 283 capabilities,
  `sha256:eeb49bec370a6bef5f30646de13ccfce86b974865aac560f51c22ece79c1b722`.
- Default policy:
  `sha256:870a92b83d3e522e4015b6bb1cabda33086906f9de1c3c36e466251ea7ed1957`.
- Producer distribution-record digest:
  `sha256:8ade9b4c5c1972b029d9393bb2586e2097cc44149a84ef8ef9ef376d634c328f`.
- Outcome: one exact fixed-order moment and its complete contraction ledger.
- Failure semantics: all scope violations are input errors with no result
  artifacts; execution failure is not a mathematical conclusion.
- Compatibility: probability semantics advance from version 2 to version 3;
  existing experimental finite-distribution request and result models are
  unchanged.

### Checker

- Exact claim: the stored result equals the complete coefficient contraction
  of the exact stored request under independent standard-real Gaussian
  semantics.
- Independence: source package `jacobian_checkers` uses only the standard
  library and is invoked in a clean process; it does not import producer code
  or Python-FLINT.
- Authorization: checker installation and policy remain operator-owned.
- Attack evidence: contract-valid changed contributions, changed metadata,
  incomplete ledgers, and fresh payload digests are rejected with `UNKNOWN`
  and no verification record.

### Evaluation

- Public cases: three answer-visible deterministic reproductions cover a
  standard sixth moment, complex cancellation, and two-variable independence.
  They are regressions, not held-out product evidence.
- Held-out comparison: protocol is frozen in the public suite as
  `READY_NOT_RUN`; no autonomous model control/treatment runs are claimed.
- Control: the catalog without the Gaussian operation.
- Treatment: the catalog with the producer and authorized
  `probability.gaussian_polynomial.moment.verify`.
- Independent oracle: a separately implemented standard-library coefficient
  contraction over generated sparse polynomials.
- Primary metrics: fixed-order correctness, checker-bound completion, false
  all-order generalization, and false `VERIFIED` rate.
- Model, prompt, and raw traces: not applicable because no model run was
  performed.
- Validation: the repository-selected unit, component, domain, composition,
  storage, process, MCP, provider, e2e, static, build, and documentation lanes
  passed; unavailable Lean and external proof executables produced only their
  declared skips.
- Decision: keep the bounded operation as experimental with its independent
  checker; do not promote or design an all-order capability from these public
  cases.

Open obligations are a genuinely held-out repeated model comparison, an
evidence-backed mixed-moment contract if recurring workflows require it, and a
separate symbolic certificate design for all-parameter identities.
