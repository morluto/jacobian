# Replayable Lean proof-state transitions

[Documentation home](../../../index.md) · [Capability surface](../../tools.md)

`lean.proof_state.apply_tactic` is the canonical implementation covering the
`lean.tactic.apply` inventory operation. Version 2 applies one bounded tactic
to an immutable, replayable Lean proof state. It does not expose or depend on
a long-lived session identifier. A replayable chain is limited to 64 tactics,
and each tactic remains limited to 1,000 characters.

## State artifact

Every state artifact binds:

- the exact proposition;
- the pinned `CORE` or `MATHLIB` environment and imports;
- Lean and Mathlib revision identifiers;
- an environment digest;
- the exact tactic prefix and generated replay source digest;
- the ordered, normalized Lean goals; and
- a state digest over all of those fields.

State artifacts are immutable and do not expire. A state is stale when its
environment digest, source digest, state digest, or cleanly replayed normalized
goals no longer match. Stale or malformed state artifacts are rejected before
the requested tactic is applied.

An invocation may start from an explicit statement and tactic prefix. That
form first creates the bound input-state artifact. Later invocations may pass
the returned state URI instead; a state URI cannot be combined with replacement
statement or prefix text.

## Clean replay

Each tactic invocation starts a clean pinned Lean REPL process. It:

1. reconstructs the exact statement and tactic prefix;
2. runs a no-op inspection transition to obtain the current goals;
3. normalizes and validates those goals against the supplied state artifact;
4. applies exactly one requested tactic; and
5. terminates the process.

No process-local proof-state number is persisted. If Lean rejects
reconstruction, the request fails without a mathematical conclusion. If
reconstruction succeeds but the tactic is rejected, the operation returns
structured diagnostics, `accepted = false`, and no successor states.

An accepted transition returns one successor-state artifact containing all
ordered goals produced by Lean. A completed successor has an empty goal list.
The transition artifact binds its input state, successor state, tactic,
diagnostics, environment digest, and replay-source digest through artifact
lineage.

## Verification boundary

Proof states and transitions have `COMPUTED` assurance only. A completed state
means that exploratory Lean reported no remaining goals for that replayed
prefix. It is not a theorem-verification record.

Every result reports `verification = UNVERIFIED` and
`verification_boundary = LEAN_CHECK_REQUIRED`. Only `lean.check`, using its
separate operator-authorized clean verification path, may verify the complete
statement and proof.

This version does not add persistent sessions, term-application semantics,
proof-axiom inspection, or trust-policy decisions.
