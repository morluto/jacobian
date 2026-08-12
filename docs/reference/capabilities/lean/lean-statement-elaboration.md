# Lean statement proposal and direct elaboration

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`lean.statement.propose` has two operations over one canonical statement
artifact:

- `PROPOSE` checks a formal statement supplied alongside an informal claim;
- `ELABORATE_PROPOSITION` directly elaborates one proposition and requires the
  informal claim to be omitted.

Both operations use `lean.statement.propose` because they share one statement
input, Lean execution boundary, artifact owner, and non-verification semantics.

## Direct proposition contract

`ELABORATE_PROPOSITION` accepts one bounded, single-expression Lean
proposition. It rejects declarations, imports, option commands, `sorry`,
`admit`, metaprogramming commands, and other structural input. Version 1
supports only the `CORE` environment, and the published request schema exposes
only that value. `MATHLIB` proof replay remains the separate, independently
registered `lean.check` boundary when its pinned project is installed.

Both statement capabilities are advertised only after Jacobian resolves and
health-checks the pinned Lean executable. If the executable is absent, has the
wrong version or commit, or cannot be resolved through the configured
toolchain, `lean.statement.propose` and `lean.statement.compare` are absent
from the installed catalog; the provider diagnostic identifies the failed
toolchain check before any invocation is attempted.

Lean elaborates the expression against expected type `Prop` with fixed
pretty-printing options. A completed result records:

- the pretty-printed elaborated core expression, when elaboration succeeds;
- structured compiler diagnostics;
- the fixed imports used to establish the environment;
- declaration names occurring in the emitted elaborated expression;
- every fixed elaboration/printing option;
- Lean version and commit identifiers; and
- an environment digest binding those identifiers, imports, and options.

The artifact is durable and content-addressed. Failed elaboration is also a
completed inspection result, but has no elaborated expression. Backend
absence, timeout, invalid input, and unsupported environments are execution
errors rather than mathematical conclusions.

## Assurance boundary

Successful elaboration means only that Lean produced a well-typed expression
of type `Prop` in the bound environment. The artifact therefore always reports:

- `semantic_scope = ELABORATION_ONLY`; and
- no verification record.

It does not show that the proposition is true, provable, equivalent to an
informal claim, or appropriate for a proof task. No proof-state or session
identity is created by this capability.
