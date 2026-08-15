---
name: verifier-evaluations
description: Design, audit, and repair fail-closed mathematical verifiers and evaluation contracts, including public schemas, frozen-input binding, task-specific witnesses, diagnostic scoring, adversarial fixtures, and Oracle validation. Use when a verifier can crash, accept malformed claims, reject equivalent witnesses, leak answers, or collapse independent diagnostics.
---

# Verifier Evaluations

Use this skill with `harbor-benchmarks`. In Jacobian, a verifier checks the
mathematical outcome of a benchmark task. `math.find` and `math.run` may be an
experimental intervention, but neither their availability nor an agent's call
trace determines task correctness. A verifier is not a parser for the
canonical solution, a grader of confidence and prose, or a recorder of which
tool calls the agent made.

## Choose the smallest checkable contract

Start from the mathematical predicate, then choose the smallest submission that
lets the verifier decide it:

- **Result only** when the verifier can replay the claim from frozen input.
- **Result plus witness** when a finite construction, counterexample, trace, or
  certificate is required for replay.
- **Formal proof** only when a supported checker accepts the language.

Put small structured mathematical certificates in `result`. Use a witness
artifact only for an external finite object that replay genuinely needs; it
must not duplicate the result or carry a natural-language explanation. Do not
add generic assurance claims, scope, completeness, limitations, or prose. If a
task exposes an independently authorized claim, publish the exact record that
authorizes it and reject a false claim. If a boundary affects score, use a
closed structured obligation ID, not a phrase, keyword count, or negation
heuristic.

Treat a typed result as the represented mathematical value, not a frozen JSON
layout or string rendering. Normalize and compare equivalent rational, scaled,
or unordered representations unless canonicalization is an explicit public
task outcome with an exact stated rule. `answer.txt` is never an authoritative
submission format.

The visible schema and instructions must describe every field, type, bound,
scope rule, and witness requirement the verifier enforces. Do not expose hidden
solutions, Oracle fixtures, or verifier implementation merely to explain the
format. A task may name the relevant Jacobian operations and their public
contracts as its experimental condition, but must not require a particular
discovery query, call order, or intermediate transcript unless that trace is
itself the mathematical object being checked. A hard task may deliberately
expose no operation that solves it today: that is a capability finding, not a
verifier defect.

## Implement total predicates

Follow this order:

```text
bounded submission → bind visible input to frozen input → exact envelope
→ semantic replay → declared witness/scope checks → reward artifact
```

At each boundary, malformed data must yield a deterministic false predicate and
a reward artifact, not an exception. Check `type(value) is int` when booleans
are invalid integers; reject non-finite numbers; validate nested shapes before
indexing or hashing; compare JSON values recursively with type-strict equality;
and accept mathematically equivalent rational, reordered, or scaled results and
witnesses when the task permits them.

Read and bound submissions and visible/frozen inputs before parsing. A witness
artifact needs a published finite bound only when its encoding or task mechanics
justify one; do not inherit a universal/default cap or create a redundant
artifact merely to add one. For declared artifacts, reject traversal, symlinks,
non-regular files, wrong cardinality, wrong digest, and content that does not
support the claimed result.

Compute mathematical correctness from the frozen verifier copy. Keep input
binding, declared witness validity, scope, and independent authorization as
separate diagnostics where they can safely be observed. Apply their hard gates
only when calculating aggregate reward. Default reward is binary: return `1`
only for a valid replayed mathematical outcome and `0` otherwise. Add partial
credit only when the public task explicitly contains independent, meaningful,
replayable mathematical subclaims; diagnostics alone never earn credit.

For a deliberate raw/strict split, raw parsing is bounded and diagnostic-only.
`load_submission()` remains the strict authoritative loader; a raw object never
bypasses public validation, witness validation, or reward gating.

## Test behavior, not implementation

Before trusting an Oracle, mutate a canonical valid submission to cover:

- malformed, missing, extra, wrong-shaped, and wrong-typed output;
- malformed or replaced visible input;
- wrong result and an alternate valid witness;
- boolean/float coercion, non-finite values, deep or oversized input;
- every declared witness failure: wrong path, digest, type, duplicate,
  traversal, symlink, or unrelated content;
- every declared structured scope or authorization failure.

Assert observable reward and diagnostics, including that the reward artifact is
written. Do not assert private helper names or mirror hidden solution text. A
large valid declared artifact should remain valid unless the published task
contract itself gives it a finite bound.

When removing a prose or envelope gate, keep a mutation that makes the
mathematical predicate false. Generic schema attacks alone do not demonstrate
that replay still rejects a wrong mathematical claim.

Natural-language proofs belong in a human-reviewed diagnostic setting unless a
formalization or executable certificate makes their key claims checkable. Never
award credit for rhetorical keywords, arbitrary nonempty text, or a preferred
proof strategy.

## Finish cleanly

After verifier, schema, Dockerfile, or task-contract changes, run focused
attacks, the selected Oracle, and the repository's planned Harbor gate. Refresh
the selected task's Dockerfile checksum after every verifier edit. When shared
support changes, migrate only deliberate task-local copies and run the affected
Oracles; do not rewrite historical snapshots.

Report the exact command, task digest, Oracle result, and any deferred
validation. Keep claims distinct: a verifier test, selected Oracle, repository
gate, and causal benchmark result are different evidence.
