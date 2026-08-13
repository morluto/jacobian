# Lean residual proof-state contracts

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

This page documents three bounded Lean proof-state contracts that extend the
[replayable proof-state transitions](lean-replayable-proof-states.md) surface
without introducing long-lived sessions, hidden planning, or trust-policy
decisions. Each contract is atomic, agent-visible, and freely composable.

## lean.term.apply

`lean.term.apply` (version 2) applies one bounded Lean term to an immutable
replayable proof state by elaborating `exact <term>` through the maintained
Lean REPL tactic protocol. It reuses the existing `lean.proof_state.apply_tactic`
clean-replay path, immutable successor-state artifact type, resource bounds,
and fail-closed boundary.

The request accepts either an explicit statement and tactic prefix (which first
materializes the bound input-state artifact) or a `state_uri` returned by a
prior proof-state operation, plus a single `term` expression. The term must be
one Lean expression: no newlines, `:=`, or forbidden commands. The adapter
constructs `exact <term>` and delegates to the proof-state adapter, so the
returned transition artifact records the elaborated tactic, the successor-state
artifact, structured goals, and the same `verification_boundary =
LEAN_CHECK_REQUIRED` semantics. It creates no verification record.

Term application does not select terms, rank successors, or prescribe proof
strategy. A completed transition still requires `lean.check` for theorem
verification.

Rejected terms use the shared Lean diagnostic model with phase
`TERM_ELABORATION` and payload-relative source `TERM`. The adapter removes its
internal `exact ` prefix from reported columns, so locations refer to the term
the caller supplied.

## lean.proof_state.inspect

`lean.proof_state.inspect` (version 1) loads an existing immutable proof-state
artifact and returns its structured goals, statement, tactic prefix, and
environment bindings without mutating or replaying it. No Lean process is
started: the returned fields are exactly those recorded on the immutable
artifact, so inspection is available whenever the artifact is available,
regardless of whether the pinned Lean runtime is installed.

The result reports `inspection = READ_ONLY_NO_REPLAY` and creates no
verification record. Stale or malformed state artifacts (whose
environment digest, source digest, or state digest no longer match) are
rejected before any field is returned.

## lean.proof_state.metavariable_fields

`lean.proof_state.metavariable_fields` (version 1) reconstructs an immutable
proof-state artifact in a clean pinned Lean process, pickles it, and asks the
pinned helper to expose typed fields through maintained Lean accessors:

- **Metavariable fields** from `MetavarDecl`: `user_name`,
  `is_user_name_anonymous`, `kind` (`NATURAL`, `SYNTHETIC`,
  `SYNTHETIC_OPAQUE`), `is_assigned`, `is_delayed_assigned`, `depth`,
  `num_scope_args`, and the rendered `target_type`. These are read via
  `MVarId.getDecl`, `MVarId.isAssigned`, and `MVarId.isDelayedAssigned`.
- **Local instance fields** from `LocalInstances`: `class_name`,
  `fvar_user_name`, and the rendered `fvar_type`, recovered from the local
  context via `LocalContext.find?`.
- **Elaboration context** from the pickled `Term.Context`: `decl_name`,
  `may_postpone`, `err_to_sorry`, `auto_bound_implicit`, `implicit_lambda`,
  `is_noncomputable_section`, `ignore_tc_failures`, `in_pattern`,
  `save_rec_app_syntax`, and `holes_as_synthetic_opaque`. Closure and
  reference fields that do not survive pickling are intentionally omitted.

The contract is only defined for states with open goals; a completed state is
rejected. Reconstruction failures, helper failures, timeouts, and errors fail
closed as non-conclusions.

### Coercion provenance limitation

`coercion_provenance` is reported as `UNAVAILABLE`. The maintained
`Lean.Meta.Coe` APIs (`expandCoe`, `getCoeFnInfo?`) operate on expressions
during elaboration and do not retain a per-metavariable coercion log on a
pickled proof state. Inferring coercions by parsing pretty-printed output is
forbidden by the repository contract model, so this operation reports the
limitation honestly rather than fabricating provenance or hand-rolling
instrumentation. The `coercion_provenance_basis` field records the reason.

## Verification boundary

None of the three contracts creates a theorem-verification record. Only
`lean.check`, using its
separate operator-authorized clean verification path, may verify the complete
statement and proof.
