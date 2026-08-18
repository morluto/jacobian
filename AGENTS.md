# Jacobian agent guide

Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow, validation,
documentation, commits, and pull requests. This file states the product choices
that changes must preserve. Consult the [product model](docs/explanation/product-blueprint.md),
[architecture](docs/explanation/architecture.md), and
[operation library reference](docs/reference/domain-operation-library.md) when
working beyond a small local change.

## What we are building

Jacobian gives agents atomic, composable tools for higher mathematics:
discovering, running, and combining typed computations to investigate
conjectures, build examples, calculate invariants, and check bounded claims.

**Jacobian's hypothesis is that mathematical reasoning benefits from an
executable vocabulary of small, exact operations.** Prefer reusable
mathematical primitives over large solvers or workflows: Jacobian supplies the
mathematical moves; the model decides how to compose them into larger solutions.

It is a **stateless MCP server for atomic, composable mathematics** with two
tools:

| Agent verb | MCP tool | Meaning |
| --- | --- | --- |
| Search | `math.find` | Find or inspect an operation. |
| Execute | `math.run` | Run one operation and return its mathematical value. |

Jacobian supplies bounded typed operations and immutable discovery; it is not a
workflow engine, planner, project manager, or durable mathematical system.
Use “operation” or “math tool,” not “product” or “provider,” for built-ins.
It is local-first: do not turn ordinary mathematical tool work into a security
project. Avoid speculative policy, tenant isolation, secret management, and
other security scaffolding in the kernel; preserve an explicit transport
boundary only when the task actually changes one.

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
semantics and types; the library does not become a provider, runtime, worker,
or second operation surface.

## Non-negotiable boundaries

- Return bounded mathematical values directly. Results may report their own exact,
  incomplete, or unknown status, but do not add generic assurance, obligation,
  verification, or completeness wrappers.
- Do not restore a workspace, SQLite/catalog overlay, artifact store,
  value-reference scheme, checker registry, runtime router, publication/replay
  record, persistence flag, or migration product. Caller-owned storage is
  outside Jacobian and should not be modeled here.
- A value that needs durability, resumability, or cross-request identity is out
  of scope. The only temporary state permitted is request-scoped data required
  for one genuinely isolated process.
- Built-in tools are explicit immutable `MathTool` tuples: discovery metadata
  plus one direct typed domain function. Do not add import-time registration,
  recursive discovery, generic registries, runtime services, installer
  callbacks, or declaration bundles. Every catalog candidate requires an
  explicit admission row in `src/jacobian/catalog/admission.py`; catalog
  construction fails closed without one (see the
  [public operation admission](docs/reference/public-operation-admission.md)
  contract).
- Keep operations composable and domain-owned. Discovery must not prescribe a
  proof strategy, next step, or stopping rule.
- Jacobian is pre-stable. Do not preserve a false request, result, or identity
  for compatibility. When a contract is broader than the implementation, reports
  a wrong mathematical value, or turns an accepted request into a host
  exception, change the contract. Do not add shims, aliases, dual fields, or
  deprecated-but-accepted shapes to keep the old behavior working.

## Implement mathematics directly

A `MathTool` is an exact mathematical instrument, not a lesson, proof recipe,
or workflow. Add an operation only when it exposes a stable bounded computation
or check that remains useful as models improve at mathematical reasoning and
notation. The model chooses what to investigate and how to compose results;
the operation returns a concrete mathematical value or certificate.

- Prefer a thin typed adapter to maintained backends such as SymPy, FLINT,
  NetworkX or Z3. They are private implementation details, not
  operation-specific providers. Do not reimplement their kernels. A public
  claim may be no broader than the implementation can establish: if the
  algorithm cannot exhaust the advertised request, shrink the request or do
  not expose the operation. A fallback, sentinel, or omitted comparison is
  not an exact invariant.
- Use a direct Python binding whenever it can perform the bounded computation.
  A subprocess needs a concrete isolation, killability, or fixed-toolchain
  reason. `lean.check` is the example: one bounded source request, temporary
  files, timeout, typed diagnostics, and no retained proof state.
- Never create a private worker protocol, worker lifecycle/registry, or durable
  worker output. SAT and SMT use their maintained Python APIs directly.
- Native public functions belong under `jacobian.math`, have explicit `__all__`,
  call typed kernels directly, and never invoke `math.run` or expose MCP,
  runtime, installation, or storage objects. They accept domain values or a
  maintained backend type that already carries the complete mathematical
  meaning (see the [native Python API](docs/reference/python-api.md) contract).

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
  under `jacobian.math`. The private root model helpers are limited to strict
  parsing and canonical scalar primitives shared by unrelated owners. Do not
  restore a global contract tree, use backend objects or JSON round trips to
  compose operations, or introduce a universal conversion layer.
- Pydantic request/result models are authoritative at operation and wire
  boundaries. Validate the complete strict request before a backend call; keep
  cross-field invariants with the owning domain model. The request must encode
  the advertised mathematical domain—bounds, positivity, completeness,
  non-degeneracy—not only the JSON shape. A request the model accepts must
  return a typed domain result; mathematical inapplicability belongs in the
  request validator or the result, not in a raised backend exception.
- Canonical decimal strings are wire values, not computation values. Use the
  canonical parse/format helpers—never direct `int()` or `str()`—and test above
  4,300 digits whenever the contract permits it.
- Construct MCP envelopes only at the final boundary. With MCP Python SDK 2.0,
  return Pydantic result models directly and use `structured_output=True`: the
  SDK derives structured output and reports request/result validation failures.
  Do not add a Jacobian error envelope, schema rewrite, or second validation
  pass. Use an explicit result only for a deliberate content projection.
- MCP owns malformed-argument and host-failure reporting. A domain result owns
  its own timeout, incompleteness, or missing-witness outcome; none is a
  mathematical conclusion by itself.

## Service and deployment

Remote requests share one immutable operation library and receive only a small
request-scoped tenant context. Deployment supplies an immutable artifact,
configuration, and health checks. Platform infrastructure—not Jacobian—owns
provisioning, TLS, supervision, rollout, rollback, secrets, and configuration.
Do not add host setup/doctor, client mutation, release-directory management,
state migration, backup/restore, or application-managed rollback. The checked-in
[`deploy/`](deploy/) files are templates; see
[remote deployment](docs/how-to/deploy-remote-mcp.md).

## Working in this repository

- Run `make check` and the named lane that owns changed behavior. Use
  `make check-external` for the fixed Lean boundary; direct Python backend work
  uses its owning domain or unit lane. CI owns the broad matrix.
- `make check-all` and `make test-full` are deliberate broad reproductions.
  Only the coordinating agent may run an exhaustive lane in a shared checkout;
  never run it concurrently with delegated validation.
- `uv run pytest <path>` is useful for focused debugging, not a substitute for
  the named lanes. `make test-process`, `make test-mcp`, and `make test-lean`
  own their specialist boundaries.
- Lean is optional; absence is a typed unavailable outcome. The ordinary
  Python backend stack is installed by `make setup`. If a non-login shell
  cannot find `uv`, add `$HOME/.local/bin` to `PATH`.
- For Harbor authoring or verifier changes, use the repository-local
  `harbor-benchmarks` skill. For recent-conjecture reliability probes, use the
  `recent-conjecture-evaluations` skill.
- A quick smoke is `uv run jacobian run integer.compute.extended_gcd --json
  '{"left":"84","right":"30"}'`; use `uv run jacobian-mcp` for local stdio
  or `uv run jacobian-remote-mcp --host 127.0.0.1 --port 8000
  --allow-anonymous` only for an explicit local remote test.
