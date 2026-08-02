# jacobian/radical-system-uniqueness-audit

Audits a corrupted multiplicity claim for a mixed-radical real system by
requiring a complete elimination, root-domain classification, and exact
reconstruction certificate.

## Benchmark classification

- **Family:** Regression
- **Primary reasoning objective:** symbolic elimination with principal-root
  domain control
- **Difficulty:** Hard (provisional; no empirical baseline yet). The task
  requires a six-root substitution, exact polynomial derivation and
  factorization, exhaustive real-root filtering, and back-substitution through
  three coupled equations.
- **Quality score:** 87/100

## Provenance

- Dataset: `INSAIT-Institute/BrokenMath`
- Revision: `5eda8c5fbd150afde41b6206b60700ab7d8e25c7`
- Config/split/row: `default` / `benchmark` / `228`
- Source ID: `german_2025_1`
- Source-row digest: `sha256:b9a10fbc445876cc412565550ef03722124d698b9ad52d0f1f6aacd0e97b823c`
- License: Apache-2.0

The public source supplies a solution, so this is a public regression rather
than held-out evidence. The frozen task does not expose that solution at
runtime.

## Selection and shortcut audit

This case adds a radical-domain failure mode absent from the current portfolio:
an algebraic elimination has roots that must be rejected using *two different*
principal-root constraints before uniqueness can be claimed. A memorized final
triple cannot pass because the submission must also provide the independently
checked elimination polynomial, complete factorization, classification of all
three distinct roots, and exact values of every original equation. Brute-force
search over triples does not establish completeness and is rejected.

Nearby BrokenMath cases were rejected when a tiny counterexample dominated the
intended reasoning, the verifier would need to encode a full olympiad proof, or
the workflow overlapped existing bounded-search and sharp-bound audits.

## Discrimination estimate

Weaker agents are expected to lose the `u=0` or `u=-1` domain obligation or
verify only the surviving triple. Stronger agents should derive the
elimination polynomial and close uniqueness. Tool-less agents can solve the
mathematics, but must maintain exact root semantics and certificate structure.

## Verification boundary

The clean-room verifier reconstructs the elimination equation, multiplies the
submitted factors, derives all distinct rational roots, checks their complete
domain classification, and validates the surviving triple in all three source
equations using exact integer arithmetic. It does not invoke an external proof
assistant and therefore permits at most `COMPUTED`.
