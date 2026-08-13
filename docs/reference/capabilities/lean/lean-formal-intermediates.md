# Lean formal intermediates

[Documentation home](../../../index.md)

- Status: Experimental contracts
- Related tutorial:
  [Retrieve a Lean theorem and check a proof](../../../tutorials/lean-declaration-discovery.md)

Jacobian exposes Lean proof construction as typed, separately invocable **math
tools**. Ordinary tools return values (declarations, proof states, premises).
Independent acceptance uses **checker tools** (`lean.check`, and proof-edit
validation bound to that checker)—separate catalog IDs, not a mode switch.

Availability depends on the pinned Lean profile and bundled-reference
installation. Inspect `operation://catalog` and call `math.find`
before using any payload below.

## Tools

| Tool | Role | Outcome |
| --- | --- | --- |
| `lean.declaration.search` | ordinary | Bounded search over public declarations |
| `lean.declaration.inspect` | ordinary | Exact declaration metadata |
| `lean.declaration.dependencies` | ordinary | Bounded type/value dependency subgraph |
| `lean.statement.propose` | ordinary | Type-check one proposed formal statement |
| `lean.statement.compare` | ordinary | Compare two formal statements without claiming equivalence |
| `lean.proof_state.apply_tactic` | ordinary | Replay a proof prefix and apply one tactic |
| `lean.retrieve.premises` | ordinary | Bounded Mathlib premise candidates with replay metadata |
| `lean.proof_edit.validate` | checker | Check an exact edited proof; bind acceptance to replay |
| `lean.check` | checker | Replay an exact proposition and proof in a pinned environment |

These tools form a portfolio, not a required workflow. Agents decide which to
compose.

## Lean diagnostics

`lean.check` version 2, `lean.proof_state.apply_tactic` version 3,
`lean.term.apply` version 2, and `lean.proof_edit.validate` version 3 share one
Lean-owned diagnostic model. A diagnostic keeps a stable `code`, processing
`phase`, severity, normalized message, optional payload-relative `source_span`,
optional `goal_index` and metavariable, and the bounded path-redacted backend
message. Source spans name the caller field (`STATEMENT`, `PROOF`, `TACTIC`, or
`TERM`) and use zero-based positions; they never expose generated temporary
file coordinates. `RUNTIME_SETUP` identifies toolchain, Mathlib-manifest, and
bounded checker-start failures; it is operational evidence and never a proof
repair verdict.

Diagnostics explain a rejected candidate without changing its mathematical
meaning. A rejected proof remains `UNKNOWN` and unverified, and a rejected
tactic creates no successor state. Operational errors and absent diagnostics
must not be interpreted as negative mathematical conclusions.

## Proof-state transitions

`lean.proof_state.apply_tactic` has two mutually exclusive request modes:

- **fresh**: `environment`, `statement`, optional `proof_prefix`, and `tactic`;
- **continuation**: the returned `state_uri`, matching `environment`, and
  `tactic`, with `statement` and `proof_prefix` omitted.

`proof_prefix` contains tactic bodies after Lean's surrounding `by`; it must not
contain `by` itself. The tool replays a fresh prefix or reconstructs the bound
immutable state before applying the new tactic, then returns:

- the exact replay source;
- rendered goals and typed goals;
- bounded local declarations for each goal;
- Lean and Mathlib runtime identity; and
- a transition artifact URI.

`completed = true` means no goals remain after this transition. It creates no
verification record. Goal count, typed-goal
indices, and completion are cross-validated so malformed backend output fails
closed.

Requests bound the number of goals, local declarations, prefix steps, and
rendered bytes. A timeout, malformed response, or rejected tactic is an
operational failure rather than evidence that the statement is unprovable.

## Premise retrieval

`lean.retrieve.premises` is available only for the pinned `MATHLIB` profile. It
replays the statement and proof prefix, invokes Mathlib's `exact?` diagnostic,
and returns a bounded ranked list of tactic candidates.

As with proof-state transitions, `proof_prefix` is a sequence of tactic bodies
after `by`. For example, use `["intro x"]`, not `["by", "intro x"]`. Invalid
prefixes fail during request validation before a Lean process starts, with a
field path and bounded validation evidence. Backend failures remain
`LEAN_RETRIEVAL_FAILED` operational non-conclusions and retain a bounded raw
backend message for repair.

Each candidate records whether its tactic replayed and which declaration names
were extracted. Name extraction is explicitly a display-text heuristic.
Retrieval is non-exhaustive and experimental; it suggests premises but does not
verify the statement or prove that omitted premises are irrelevant. The
`goal_context_digest` binds the result to the exact replayed context.

## Dependency subgraphs

`lean.declaration.dependencies` starts from one exact declaration and traverses
constants referenced by elaborated types and values. `max_depth` and
`max_nodes` bound the traversal. Returned edges distinguish `TYPE` and `VALUE`
references.

Inspect `closure_complete`, `node_budget_exhausted`, and `frontier` together.
A non-empty frontier identifies returned nodes whose dependencies were not
fully expanded. A partial graph is useful context, not a complete dependency
claim. The durable-operation result uses `result_uri` for the graph artifact
and includes the complete graph under `preview`; the artifact binds the query
and pinned `environment_digest`.

## Statement operations

`lean.statement.propose` checks that one proposed statement elaborates in the
selected environment. Elaboration establishes that the expression is a valid
Lean proposition; it does not establish its truth or its correspondence to an
informal claim.

`lean.statement.compare` exposes bounded syntactic and elaborated comparison
data for two statements. Similarity or matching structure does not certify
logical equivalence. If equivalence matters, formulate and check that relation
as an explicit theorem.

## Exact proof edits

`lean.proof_edit.validate` accepts a statement, an original proof, and a
different edited proof. It rejects unchanged edits and disallowed holes before
checker invocation, materializes the claim, candidate, certificate, and unified
diff, and calls the operator-authorized Lean checker.

The result may set `accepted = true` only when it also contains a verification
record URI. Checker timeout, cancellation, rejection, malformed output, or
runtime drift leaves the edit unaccepted.

Proof repair remains an agent-owned composition. Use proof-state inspection and
premise retrieval to construct an exact edit, then submit that edit to
checker-backed validation. Generation, tactic success, or an empty displayed
goal state cannot accept the edit without the bound verification record.

## Assurance summary

Declaration discovery, dependency extraction, statement elaboration,
comparison, tactic transitions, and premise retrieval return `COMPUTED`
evidence. Their pinned environment identity makes the computation replayable
but does not grant mathematical assurance.

`lean.check` and accepted proof edits may return `VERIFIED` because an
operator-authorized checker independently replays the exact bound proposition
and proof. No retrieval rank, elaborator success, empty goal display, or model
answer can substitute for that record.
