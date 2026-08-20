# Jacobian agent guide

This file states the product choices and repository invariants that changes must
preserve. Follow [CONTRIBUTING.md](CONTRIBUTING.md) when selecting validation,
changing documentation, preparing commits or pull requests, or working on
releases and evaluations. Follow the
[product model](docs/explanation/product-blueprint.md) and
[architecture](docs/explanation/architecture.md) when changing product scope,
ownership, or the execution path.

## What we are building

Jacobian gives agents atomic, composable tools for higher mathematics:
discovering, running, and combining typed computations to investigate
conjectures, build examples, calculate invariants, and check bounded claims.

**Jacobian's hypothesis is that mathematical reasoning benefits from an
executable vocabulary of semantically scoped, bounded operations.** Prefer
reusable mathematical primitives over large solvers or workflows: Jacobian
supplies the mathematical moves; the model decides how to compose them into
larger solutions.

Atomicity is semantic: an operation establishes one stable, reusable
mathematical postcondition. It need not have a small or simple implementation.

It exposes two MCP tools:

| Agent verb | MCP tool | Meaning |
| --- | --- | --- |
| Search | `math.find` | Find or inspect an operation. |
| Execute | `math.run` | Run one operation and return its mathematical value. |

Jacobian supplies bounded typed operations and immutable discovery. Use
“operation” or “math tool,” not “product” or “provider,” for built-ins. It is
local-first: ordinary mathematical tool work should stay focused on mathematics;
preserve an explicit transport/security boundary only when the task actually
changes one.

The ordinary execution path is deliberately this small:

```text
math.run(operation ID, JSON)
  -> select the immutable declaration
  -> parse its Pydantic request once
  -> call the domain-owned function
  -> return its concrete typed mathematical result
```

The domain function may use a maintained library as a private computational
engine; prefer an established backend over hand-rolling a kernel whenever it
can perform the bounded computation. Jacobian owns the public mathematical
semantics and types.

## Non-negotiable boundaries

- Return bounded mathematical values directly. Results may report their own exact,
  incomplete, or unknown status, but do not add generic assurance, obligation,
  verification, or completeness wrappers.
- **Anti-regression:** keep the kernel stateless; the caller owns composition and
  durable state. Internal temporary state is request-scoped and exists only when
  one bounded external call genuinely requires it.
- Built-in tools are explicit immutable `MathTool` tuples: discovery metadata
  plus one direct typed domain function. Every catalog candidate requires an
  owner-local admission decision in its mathematical domain's `_admission.py`
  module; `jacobian.catalog.admission` owns the shared policy types and
  fail-closed validation (see the
  [public operation admission](docs/reference/public-operation-admission.md)
  contract).
- Before adding a public operation, establish that the missing capability is an
  operation gap rather than a discovery, representation, interoperability,
  contract, scale, backend, or reasoning failure.
- Keep operations composable and domain-owned. Discovery must not prescribe a
  proof strategy, next step, or stopping rule.
- Jacobian is pre-stable. When a request/result contract is broader than the
  implementation, reports a wrong mathematical value, or turns an accepted
  request into a host exception, change the contract rather than preserving the
  old shape through compatibility machinery.

## Implement mathematics directly

> **Jacobian owns the contract; the backend owns the kernel.**

```text
public request
  -> canonical Jacobian value
  -> complete mathematical and work admission
  -> private backend adapter
  -> exact backend result
  -> invariant validation
  -> canonical Jacobian result
```

- Public contracts use canonical mathematical values, not backend expressions
  or ambient contexts.
- Backends never define the accepted public domain through runtime exceptions.
- Intentional changes of ring, field, parent, or axis require explicit typed
  maps; implicit coercion is forbidden.
- Exact results retain all information needed for reconstruction and downstream
  composition.

A `MathTool` is a bounded mathematical instrument, not a lesson, proof recipe,
or workflow. Add an operation only when it exposes a stable bounded computation
or check that remains useful as models improve at mathematical reasoning and
notation. The model chooses what to investigate and how to compose results;
the operation returns a concrete mathematical value or certificate.

- Prefer a thin typed adapter to an appropriate maintained mathematical library.
  Treat any library, solver, or external tool as a private computational engine;
  do not reimplement its kernel. Research established options before writing a
  custom algorithm. A public claim may be no broader than the implementation
  can establish. If the engine cannot exhaust the advertised request, narrow
  the request or do not expose the operation. A fallback, sentinel, or omitted
  comparison is not an exact invariant.
- Use a direct Python binding whenever it can perform the bounded computation.
  A subprocess needs a concrete isolation, killability, or fixed-toolchain
  reason. `lean.check` is the example: one bounded source request, temporary
  files, timeout, and typed diagnostics.
- Native public functions belong under `jacobian.math`, have explicit `__all__`,
  call typed kernels directly, and accept domain values or a maintained backend
  type that already carries the complete mathematical meaning (see the
  [native Python API](docs/reference/python-api.md) contract).

### Mathematical boundedness is a proof obligation

Jacobian is a library of mathematical instruments for agents doing high-level
mathematics and investigating conjectures. Treat every operation as a
trust-bearing function: do not paper over an unproved algorithm or backend
limit with a sentinel, truncation, post-hoc conversion failure, or optimistic
contract. Separately bound the accepted input, algorithmic work and
intermediates, and exact result or certificate. Derive budgets from the
mathematics before backend expansion; use explicit, named, tested conservative
domains when necessary, and narrow the domain or change the typed result when
the claim cannot be established. Add known-answer, boundary, adversarial, and
defining-invariant tests.

## Types and transport

- Domain values, request/result models, and declarations live with their owner
  under `jacobian.math`. Compose operations through typed mathematical values.
  Follow the [native Python API](docs/reference/python-api.md) when changing
  exported Python functions or values.
- Validate the complete mathematical request before invoking a backend. Every
  accepted request must return a typed result rather than expose a backend or
  host exception. Follow the
  [operation library](docs/reference/domain-operation-library.md) when changing
  an operation contract or implementation.
- For every public string field that carries mathematical syntax, name its
  grammar and prove that parsing is non-evaluating. Caller input must never
  reach `sympify`, `parse_expr`, `eval`, `exec`, or an evaluator generated by
  `lambdify`.
- For every backend call, document the backend's accepted mathematical domain
  and encode that domain in the request model before invocation.
- For every exact decomposition, certificate, or authoritative derived value,
  state its reconstruction or defining invariant and test that invariant.
- Keep mathematical results separate from transport failures. Timeout,
  incompleteness, unavailable execution, and missing witnesses do not establish
  mathematical conclusions. Follow the
  [tool reference](docs/reference/tools.md) only when changing MCP projection or
  transport behavior.

## Service and deployment

Keep mathematical execution stateless and deployment responsibilities outside
the operation library. Follow the
[remote deployment guide](docs/how-to/deploy-remote-mcp.md) only when changing
authentication, configuration, health checks, deployment templates, or remote
service behavior.

## Working in this repository

- Preserve unrelated work in a shared checkout. Agents must not concurrently
  switch branches, stage, commit, clean, rewrite history, or edit overlapping
  paths.
- Run `make check` and the named lane that owns changed behavior before
  handoff. Only the coordinating agent may run exhaustive validation in a
  shared checkout.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [testing strategy](docs/reference/testing-strategy.md) when selecting
  specialist validation or preparing a contribution.
