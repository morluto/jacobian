# Finite probability operations

[Documentation home](../../../index.md)

- Status: Experimental contracts
- Domain: `probability`
- Producer backend: pinned Python-FLINT exact rational arithmetic
- Checker backend: Python standard-library `Fraction`, isolated from the producer

Jacobian's finite-probability bundle exposes exact, bounded operations on
explicit rational distributions. Each distribution has at most 256
strictly increasing, distinct rational support values, nonnegative rational
masses, and total mass exactly one. Inputs are completely validated before
backend execution or artifact writes.

## Operations

| Capability | Atomic outcome | Inspectable ledger |
| --- | --- | --- |
| `probability.finite_distribution.raw_moment.compute` | One exact raw moment of order 0 through 128 | Per-atom power and weighted contribution |
| `probability.finite_distribution.event_probability.compute` | Exact mass of an explicit support subset | Every selected source atom |
| `probability.finite_distribution.condition.compute` | Normalized distribution on one positive-mass explicit event | Source and conditioned mass per selected atom |
| `probability.finite_distribution.pushforward.compute` | Exact distribution under one total finite lookup map | Source, target, and transported mass per atom |
| `probability.finite_distribution.convolution.compute` | Sum distribution for two explicitly independent finite variables | Every bounded product-measure pair |

Event inputs contain canonical support values rather than executable
predicates. Pushforward inputs must cover every source atom exactly once in
canonical source order. Convolution is capped at 4,096 pairs, and validates
the size and rational-growth bounds before computation.

Conditioning on an exact zero-mass event returns
`FINITE_CONDITIONING_ZERO_MASS` as a non-applicability outcome, with no result
artifacts and no mathematical conclusion. Validation errors, unavailable
providers, timeouts, and execution errors likewise do not establish a
probability claim.

## Independent verification

Each producer has a per-producer verifier: `probability.finite_distribution.raw_moment.verify`,
`probability.finite_distribution.event_probability.verify`,
`probability.finite_distribution.condition.verify`,
`probability.finite_distribution.pushforward.verify`, and
`probability.finite_distribution.convolution.verify`. The operator-authorized
checker runs in a clean process and uses only standard-library rational
arithmetic for the mathematical replay. These inline producers accept a typed
input and candidate; verification materializes and binds both artifacts with
the semantics, candidate digest, witness format, and complete contribution
ledger.

The checker replays normalization, event membership, conditional division,
pushforward aggregation, or every convolution pair as appropriate. A producer
result remains `COMPUTED`; only an accepted, locally recorded checker run for
that exact bound candidate returns `VERIFIED`. Malformed artifacts, substituted
sources, altered ledgers, missing convolution pairs, and mathematically false
but schema-valid candidates are rejected without a verification record.

This finite-distribution foundation does not cover predicate-defined events,
approximate distributions, all-order Gaussian identities, Markov chains,
graph reliability, or probabilistic inference strategy. The same probability
bundle separately exposes one
[bounded Gaussian polynomial moment](../polynomial/gaussian-polynomial-moments.md) and one
[small exact graph reliability](../graphs/graph-reliability.md) outcome.
