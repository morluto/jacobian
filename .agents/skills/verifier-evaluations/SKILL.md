---
name: verifier-evaluations
description: Design, audit, or repair mathematical benchmark verifiers, submission contracts, and scoring.
---

# Verifier Evaluations

A verifier decides a benchmark's mathematical predicate from frozen input and
a bounded submission. It does not grade prose, confidence, tool use, or equality
with one preferred solution. Use `harbor-benchmarks` only when the task also
needs Harbor packaging, environment changes, or execution guidance.

## Establish the contract

Choose the smallest checkable submission: a typed result, a result with a
necessary finite witness, or a supported formal proof. Accept mathematically
equivalent representations unless canonicalization is an explicit task outcome.
The visible instruction and schema must describe every enforced condition and
must not leak the solution or derived conclusions.

For a new or changed submission shape, read
[submission contracts](references/submission-contracts.md), including witness
selection, schema reductions, and independent result fields. Keep instruction,
schema, verifier, gold, public contract, and host tests consistent when changing
that shape.

## Replay and score

Bound input before parsing; enforce exact types and shapes before computation.
Replay mathematics against the frozen verifier copy. Malformed submissions must
produce a deterministic false predicate and reward artifact, not a host exception.
Input binding, mathematical correctness, and declared witness validity remain
separate diagnostics where independently observable; required gates combine in
reward. Default reward is binary. Partial credit requires explicitly declared,
independent mathematical subclaims.

For implementation, diagnostic binding exceptions, artifact/path handling, or
regression tests, read [replay and attacks](references/replay-and-attacks.md).
Preserve alternate valid witnesses and a discriminating wrong mathematical
claim alongside malformed-input cases. Natural-language proofs need human
review unless formalization or an executable certificate makes them checkable.

## Complete the repair

Run focused behavioral attacks, the selected Oracle, and the repository's
planned Harbor gate when the verifier or task contract changes. Refresh selected
Dockerfile checksums after verifier edits through the task preparation workflow.
Shared support changes require the affected task-local copies and Oracles;
historical snapshots remain unchanged.

Report the actual command, task digest, Oracle result, and deferred validation.
A verifier test, Oracle pass, repository gate, and causal benchmark comparison
establish different evidence.
