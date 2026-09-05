# Replay and adversarial validation

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
a symlinked solution root, non-regular files, wrong cardinality, wrong digest,
and content that does not support the claimed result.

Compute mathematical correctness from the frozen verifier copy. Keep input
binding, declared witness validity, scope, and independent authorization as
separate diagnostics where they can safely be observed. Apply their hard gates
only when calculating aggregate reward. Default reward is binary: return `1`
only for a valid replayed mathematical outcome and `0` otherwise. Add partial
credit only when the public task explicitly contains independent, meaningful,
replayable mathematical subclaims; diagnostics alone never earn credit.
`load_submission()` may refuse an unbound workspace input; that must not zero
`correctness` or `witness_validity` when those diagnostics are independent.
Parse without requiring binding, replay against frozen tests input, and AND
binding only into `reward`.

Declare diagnostic exceptions such as `input_binding_decoupled` in the task’s
closed `tests/verifier_contract.json`; do not use a global task-name registry.

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
