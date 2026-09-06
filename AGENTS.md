# Jacobian agent guide

Jacobian gives agents atomic, composable tools for higher mathematics.
`math.find` discovers immutable operation declarations; `math.run` executes one
bounded typed computation. The caller owns reasoning, composition, and durable
state. Use “operation” or “math tool,” not “product” or “provider,” for built-ins.

## Product invariants

- Atomicity is semantic: one operation establishes one stable, reusable
  mathematical postcondition. Alternative algorithms remain private kernel
  choices. Discovery describes mathematics, not a proof strategy or call order.
- Keep the kernel stateless. Temporary execution state is request-scoped.
  Return mathematical values directly, without generic assurance, verification,
  obligation, or completeness wrappers. Operational non-completion never
  establishes a mathematical conclusion.
- Return ordinary exact values without certificates, source digests, or
  `verified` flags by default. Include a witness only for the operation's
  mathematical purpose. A consumer checks an authored relation only when its
  result relies on it; it does not verify a value's entire producer history.
- Built-ins are immutable `MathTool` tuples in owner-local `_tools.py` manifests
  under `jacobian.math`. Manifest presence is publication; catalog construction
  fails closed on malformed declarations and duplicate IDs.
- Each mathematical value has one domain-owned canonical type. Producers and
  consumers compose unchanged through serialization, including empty and
  degenerate values. Exact results retain reconstruction and ambient context;
  changes of ring, field, parent, or axes require explicit typed maps.
- Admit mathematical work, intermediate growth, and exact output before backend
  expansion. Compute semantic admission once after canonicalization. Validators,
  result construction, worker decoding, and transport must not replay computed
  mathematics. Caller-supplied claims require their own admitted domain check
  when a consumer relies on them. Moving checks out of constructors requires
  migrating every dependent consumer, not deleting the checks.
- **Scale first:** probe a motivating rejected request, improve the estimate,
  representation, algorithm, or backend, and retain cheaply executable cases as
  accepted regressions. Measurements do not replace a sound bound. A remaining
  limitation must be reported, not presented as a completed scale repair.
- Prefer maintained mathematical backends through thin typed adapters. Jacobian
  owns the accepted contract; backend exceptions cannot define it. A subprocess
  needs a concrete isolation, killability, or fixed-toolchain reason. All
  mandatory phases share the request's deadline and work accounting; wall time
  is a safety limit, not mathematical evidence or a universal short timeout.
- Mathematical syntax has a named, non-evaluating grammar. Caller strings must
  never reach `sympify`, `parse_expr`, `eval`, `exec`, or generated evaluators.
- Jacobian is pre-stable: repair an unsound contract rather than preserving it
  through compatibility machinery. Diagnose operation, representation,
  interoperability, discovery, contract, scale/backend, and reasoning gaps
  before proposing a new public operation.

## Read for the change

Use the relevant sections; these links are not a prerequisite reading stack.

| Change | Authority |
| --- | --- |
| Product scope, ownership, execution path | [Product model](docs/explanation/product-blueprint.md), [architecture](docs/explanation/architecture.md) |
| Operation implementation or contract | [Operation library](docs/reference/domain-operation-library.md) |
| New public operation | [Public operation admission](docs/reference/public-operation-admission.md) |
| Native functions or mathematical values | [Python API](docs/reference/python-api.md), [schemas and interoperability](docs/reference/value-interoperability.md) |
| Backend adapter or child worker | [Backend contract](docs/reference/mathematical-backends.md) |
| MCP projection or transport | [Tool reference](docs/reference/tools.md) |
| Authentication, health, or deployment | [Remote deployment](docs/how-to/deploy-remote-mcp.md) |
| Validation, docs, contributions, or evaluations | Relevant section of [CONTRIBUTING.md](CONTRIBUTING.md) |

For mathematical changes, establish independent correctness evidence, invariant
ownership across the execution path, and useful accepted boundaries. The
[contributor quick path](CONTRIBUTING.md#contributor-quick-path) and
[testing strategy](docs/reference/testing-strategy.md) define the evidence and
owning lanes. Use real mathematical behavior, not fakes or source-text assertions,
for correctness. Reproduce a reported backend or admission defect before repair.

## Shared work and completion

Preserve unrelated work. Agents must not concurrently switch branches, stage,
commit, clean, rewrite history, or edit overlapping paths. Parallel PR writers
need isolated worktrees, distinct branches, and one active writer per branch;
record and check issue/PR claims before implementation. Only the coordinator
runs exhaustive validation in a shared checkout.

Before a first public-operation push, search open PRs for its ID. Identify any
superseded contract in the PR description. After catalog-conflict resolution,
run catalog conformance and compare the final diff with intended public symbols.
Fetch immediately before pushing and inspect a changed head. Never push to a
merged or closed PR head; use a follow-up branch.

Complete authorized implementation, relevant validation, and repairs caused by
the change before handing back. Rerun only checks invalidated by later edits.
Use the contributor guide's affected checks; broad validation is conditional.
Report remaining limitations and checks that could not run. Existing user
authorization persists; local work does not imply permission for external writes.
