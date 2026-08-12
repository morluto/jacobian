# Powerful-number decision

[Documentation home](../../../index.md)

`integer.decide.powerful` decides whether one positive integer is powerful and
preserves the complete prime factorization used for the decision. The
capability is an atomic predicate, not a range search or a proof of a statement
about consecutive powerful numbers.

## Contract and semantics

The request contains:

- `value`: one canonical positive decimal integer string of at most 256
  characters; and
- `resource_budget.wall_seconds`: an integer from 1 through 30, defaulting to
  5 seconds.

The result contains:

- `semantics_version`:
  `powerful-number.prime-exponents-at-least-two.v1`;
- `is_powerful`: whether every prime in the complete factorization has
  exponent at least two;
- `factors`: the canonical ascending prime-power factorization; and
- `violating_primes`: exactly the ascending factor bases whose exponent is
  less than two.

The value `1` is powerful: its factorization and violating-prime list are both
empty. Zero and negative inputs fail full request validation before any
operation artifacts are written.

The result schema binds the boolean to the witness: factor bases must be
strictly increasing and greater than one, `violating_primes` must exactly
match the factors with exponent below two, and `is_powerful` must be true
exactly when that list is empty.

## Execution and assurance

The capability extends the explicit number-theory factorization bundle and
runs SymPy `factorint` inside the existing isolated, resource-bounded worker.
A timeout, resource exhaustion, malformed worker response, cancellation, or
worker error is a non-conclusion and never becomes `false`.

Successful execution has `COMPLETE` completeness for the single input and
`COMPUTED` assurance. The producer does not verify itself.

## Independent verification

The operator-authorized `integer.powerful.verify` capability accepts the
stored result URI and independently replays the exact claim with Python-FLINT.
The checker imports neither SymPy nor the producer or worker modules. A
successful replay creates a verification record and may promote that exact
stored result to `VERIFIED`.

The checker fails closed unless all of these obligations hold:

| Obligation | Independent check |
| --- | --- |
| Artifact binding | Bind the exact input, result, semantics, schemas, lineage, witness envelope, and checker identity. |
| Input domain | Require the canonical positive decimal and exact bounded producer-budget shape. |
| Result fields | Require the exact semantics version, strict boolean, factor list, and violating-prime list. |
| Complete factorization | Reconstruct the input product and compare every canonical prime power with Python-FLINT factorization. |
| Predicate | Recompute whether every independently replayed exponent is at least two. |
| Violations | Require exactly the ascending prime bases whose replayed exponent is below two. |
| Runtime | Require the operator-authorized checker source and pinned Python-FLINT/FLINT runtime. |

A malformed, substituted, incomplete, noncanonical, or mathematically false
candidate is `REJECTED` with conclusion `UNKNOWN`; it is never converted into
a contrary theorem. Checker unavailability, timeout, cancellation, or error is
also non-conclusive.

## Scope limits

Each invocation decides one integer. Repeating it over a finite caller-chosen
sample does not certify exhaustive coverage of a range or prove an unbounded
conjecture. Those claims require separately scoped enumeration or formal
certificate evidence.
