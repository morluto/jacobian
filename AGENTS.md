# Repository Guidelines

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, documentation,
commits, and pull requests. This file lists only Jacobian-specific constraints.
Load the [product model](docs/explanation/product-blueprint.md),
[architecture](docs/explanation/architecture.md), or
[tool reference](docs/reference/tools.md) when needed. For built-in mathematical
operations, also use the
[domain operation library reference](docs/reference/domain-operation-library.md).

## Product Constraints

Jacobian is an **MCP server with two tools for atomic mathematics**. It is not a
workflow engine: the agent owns decomposition, sequencing, and stopping.

| Agent verb | MCP tool | Meaning |
| --- | --- | --- |
| Search | `math.find` | Find or inspect math tools (IDs, schemas, examples). |
| Execute | `math.run` | Run **one** tool → **mathematical value**. |
| Inventory | `operation://catalog` | Full catalog when needed. |

See [product model](docs/explanation/product-blueprint.md) and
[Search and execute](docs/explanation/architecture.md#search-and-execute).
Not a required sequence: agents may run a known ID, search first, or re-find
mid-investigation.

**Results are math-first.** Operations return calculations (GCD, matrix, path,
factors, …) plus execution status. Do not add generic assurance, completeness,
scope, or obligation knobs to ordinary results; bounded status belongs in the
domain result that defines it. `lean.check` is a one-shot source-checking
operation, not a general checker framework.

- Server: typed contracts, resource bounds, immutable discovery, and the final
  MCP projection.
- Model: representation, which tools to run, how to compose values, when to
  stop.
- Discovery must not become a hidden planner (“recommended next step”).
- Evaluations reward correct math, useful intermediate values, safety, and
  efficiency—not a fixed tool-call sequence.

**Naming.** Use **math tool** / **operation** throughout Jacobian-owned code,
catalogs, documentation, tests, and wire contracts.

**Lifecycle vocabulary.** A built-in operation is a typed mathematical
function shipped by Jacobian. Ordinary inline operations are live
`InlineOperation` declarations; `math.find` and `math.run` load them with no
state directory. Serving neither discovers nor installs operations. Reserve
**installation** for Jacobian itself and the fixed Lean environment used by
`lean.check`. Ordinary maintained libraries such as SymPy, NetworkX, FLINT,
and Z3 are private math backends, not operation-specific providers.

Tools stay atomic, searchable, and freely composable. No prescribed proof
strategy or stopping criteria in discovery, ranking, prompts, or adapters.

Design against the portfolio. Reuse typed values; prefer composable
primitives over paper-shaped mega-tools. Domain-owned tool IDs over generic
verb taxonomies or new top-level MCP tools.

Prefer thin adapters to maintained mathematical systems. Wrap Jacobian's
semantics, not an entire backend API: one public function has one canonical
semantic input, validates the domain it promises, and delegates the algorithm
through a private backend module without silently changing domains or parents.
Pin versions when reproducibility, certificates, or verification depend on
them.
Do not reimplement proof kernels, elaborators, tactic engines, solver engines,
computer algebra algorithms, or graph canonicalization when a maintained
backend provides the needed operation.

The supported native Python API lives only under `jacobian.math`. Keep its
public modules deliberately small, declare their supported symbols with
explicit `__all__` values, and cover namespace and import isolation in the
public-API tests. Do not re-export domain APIs from the root `jacobian`
namespace. Native functions accept and return Python or maintained
backend-native values and call typed mathematical kernels directly; they must
not invoke `math.run`, construct an operation runtime, or expose MCP,
artifact, provider-loading, or installation objects.

### Mathematical interoperability

Operations interoperate through shared, typed domain values—not
backend-specific objects, JSON round-trips, or wire encodings. Domain values
live beside their public functions under `jacobian.math.<domain>.values`;
`jacobian.contracts` is limited to genuinely cross-domain passive primitives.
Add explicit domain-owned conversions when representations differ. Cover
producer-to-consumer compatibility and canonical or backend-native round trips
in tests. Architecture checks must reject internal JSON round-trips and unsafe
canonical conversions.

Domain values own provider-independent identity. Private backend conversions
connect them to computational values. Every public mathematical function has
one canonical semantic input type; use a maintained backend type only when it
already carries complete semantics. Do not add a universal backend wrapper,
automatic coercion framework, generic conversion language, or second semantic
type system above maintained libraries.

Canonical decimal strings are wire values, not computational
values. Use the canonical conversion API before calling backends or constructing
results. Do not directly apply `int()` or `str()` to canonical components or
change `sys.set_int_max_str_digits()` as a workaround. Keep backend coercion in
thin adapters, and test above 4,300 digits whenever the contract permits it.

Keep Pydantic models authoritative at operation and wire boundaries. Domain
implementations and operation factories must preserve
their concrete request and result types: do not accept
`Callable[[ContractModel], ContractModel]` or cast a validated request back to a
domain model. A bounded operation records exact, incomplete, or unknown status
in its domain result instead of adding generic completeness or obligation
wrappers. When a native API and an operation expose the same outcome, share one
typed mathematical kernel and use explicit domain-owned conversions rather
than duplicating the mathematics or introducing a generic conversion framework.

Shared abstractions require two surviving production paths and must delete the
older duplication in the same change. An ordinary operation should need at most
one public domain function, one request model when necessary, one rich result
model when necessary, and one semantic operation declaration. Bounded results
are returned inline; do not add publication or retention layers.

Do not introduce pass-through facades to make a dependency graph look cleaner.
An aggregate may coordinate a public lifecycle or transaction, but it must not
mirror every collaborator method, bounce callbacks back through itself, or keep
private forwarding methods solely for tests. Call the concrete owner directly;
extract a shared abstraction only when two production paths replace their older
implementations in the same change.

Composition roots retain only resources needed after construction. Do not build
or return nested installation reports, provider bags, or phase-result mirrors
that production immediately discards. Keep an installation result only when a
later production phase consumes that exact typed fact.

Construct wire envelopes only at the final MCP projection. Mathematical
functions and operation executors return their owned typed values or terminal
states; they do not construct `OperationResult`.

At the MCP boundary, prefer MCP Python SDK 2.0 high-level typed returns. Return
Pydantic result models directly and let the SDK derive the output schema,
validate results, and populate `content` and `structured_content`. Use an
explicit `CallToolResult` only for a deliberate text projection. Set
`structured_output=True` so unsupported return annotations fail during
registration.

Return bounded mathematical values inline. Jacobian has no artifact, resource,
or persistence product; do not add retention flags, resource links, replay, or
publication to an operation.

Built-in mathematical operations belong in explicit declaration modules. Do not
add global operation registries, recursive package discovery, import-time
registration, or mechanical wrappers for backend functions.

Declaration modules export immutable operation tuples. Do not add bundle
objects, installer callbacks, runtime services, storage collaborators, or
dependency-resolution policy around them. An operation may call a typed private
computational backend; that backend is part of execution, not application
lifecycle.

Keep availability, recommendations, and compatibility separate. Experimental
contracts may break between versions; compatibility applies only to supported
versions.

Follow the
[ownership model](docs/explanation/product-blueprint.md#ownership-model).
Keep strategy out of the kernel and semantics out of generic contracts.

## Bounded-result rules

- Treat `TIMEOUT`, `CANCELLED`, `ERROR`, incomplete enumeration, and failure to
  find a witness as non-conclusions.
- Keep execution status, input validity, and the domain mathematical conclusion
  separate.
- Do not promote an evaluator score, solver status, model answer, or search
  result beyond the conclusion stated by its typed domain result.

## Repository Gotchas

- Before final validation, use `make check` plus the named lane that owns the
  changed behavior on the final tree (`make check-external` for Lean/Mathlib,
  `make test-provider` for optional or maintained Python providers). In a
  shared checkout, agents must own disjoint paths and must not switch
  branches, stage, commit, clean, or rewrite shared files until their work
  is integrated.
- Jacobian is pre-stable. Current reference documents and the installed catalog
  define the supported surface; they do not order operation research.
- Validate the complete Pydantic request model before preflight, provider calls,
  and computation. This includes relationships among
  individually valid fields: parents, characteristics, presentations, axes,
  bases, labels, and bound identities must agree where the operation requires
  them. JSON Schema supports discovery; it does not replace cross-field model
  validation. Exercise incompatible-but-individually-valid values through the
  serialized installed-operation boundary and assert an invalid-request result
  with no execution.
- Parse agent-supplied JSON strictly into the owning Pydantic request model;
  advertised integers must not accept numeric strings or other coercions. An
  adapter prepares that typed request before provider readiness and executes
  only the prepared value. Strict transport encoding is lossless: do not reduce
  rationals, normalize Unicode, or otherwise repair semantic input before the
  owning model validates it. At the final projection, require the published
  Pydantic model to match the installed output contract before serializing it.
- Mathematical inputs are not presumed confidential. Public diagnostics should
  expose a stable domain reason, path, limit, and recovery direction—not
  arbitrary rejected values, which may be unbounded or user-controlled. This
  projection must not add another validation pass.
- A `COMPLETED` bounded operation may return a domain result marked `UNKNOWN` or
  `INCOMPLETE`. Execution completion alone does not establish optimality or a
  mathematical conclusion.
- Results carry values and execution state only; they never carry artifact or
  verification-record references.
- A missing maintained Python backend is a broken installation. `lean.check`
  reports a bounded provider failure when the fixed Lean environment is absent.
- Keep `deep_review.md` local; it is ignored and is not design source material.
- Keep worked cases in reference scenarios and benchmarks.

## Agent Workflow Entry Points

Operation work remains agent-directed and is not coupled to a mandatory
development workflow. For Harbor task authoring and verifier changes, use the
repository-local [`harbor-benchmarks`](.agents/skills/harbor-benchmarks/SKILL.md)
skill and its exact task validation path. Control/treatment model evaluations
are explicit operator-run evidence exercises, not routine development gates.
For source-grounded held-out reliability probes based on recently resolved
conjectures, use
[`recent-conjecture-evaluations`](.agents/skills/recent-conjecture-evaluations/SKILL.md).

For remote MCP operation, use
[Deploy the remote MCP server](docs/how-to/deploy-remote-mcp.md) and the
checked-in files under `deploy/`. They define the reproducible systemd, Caddy,
Tailscale Funnel, smoke, restart, and rollback baseline. Files under `tmp/` are
ignored host-local evidence and are never deployment source of truth. Compare
the MCP-advertised package version with the selected checkout during every
redeploy; an unchanged catalog does not prove that the backend restarted.

## Cursor Cloud specific instructions

This is a Python 3.12 project managed with `uv`; the base image ships Python 3.12
and Node but not `uv`. The startup update script installs `uv` (to
`~/.local/bin`, added to `PATH` via `.bashrc`/`.profile`) and runs
`uv sync --locked --dev`. The base dependency set includes the pinned SymPy,
NetworkX, Python-FLINT, Z3, and cvc5 providers. Standard dev, test, lint, and
build commands live in the `Makefile` (`make help`) and `CONTRIBUTING.md`; use
those rather than duplicating them.

Non-obvious caveats:

- If a fresh non-login shell can't find `uv`, run `export PATH="$HOME/.local/bin:$PATH"`.
- Lean is optional. When its fixed environment is absent, `lean.check` returns
  its typed unavailable outcome. SAT and SMT use their maintained Python APIs;
  there are no proof-artifact commands or solver-worker installations.
- `make test-unit` is the cheap unit lane. `make quick` adds lint; `make check`
  adds lint and typecheck. `make check-all` explicitly reproduces the Lean-free
  ordinary CI matrix. Use `make test-full` only
  for an explicit exhaustive local reproduction; it takes this worktree's
  exhaustive validation lock (`make validation-status`). Default `uv run pytest` does
  not collect Lean, storage, process, or MCP; use the matching `make test-*`
  target for those trees. Never run bare `uv run pytest` as a substitute for
  the complete specialist matrix.
- Only the coordinating agent may start an exhaustive test lane. Never delegate
  one to a parallel agent sharing the host. Before an exceptional broad run,
  inspect active processes for pytest jobs from this checkout and stop or wait
  for them; concurrent runtime/store/subprocess suites turn per-test timeouts
  into a host-contention detector rather than useful failure evidence.
- Concurrent suites can still contend for CPU, process startup, and native
  libraries. Reproduce a timeout in the owning focused test before treating it
  as a product defect.
- Quick end-to-end smoke of the product surface: `uv run jacobian run
  integer.compute.gcd --json '{"left":"84","right":"30"}'`,
  `uv run jacobian-mcp` for one local stdio server, or
  `uv run jacobian-remote-mcp --host 127.0.0.1 --port 8000 --allow-anonymous`
  for an explicit remote test host. Remote hosting requires `--allow-anonymous`
  or `--auth-tokens-file`; those options are intentionally absent from the local
  entry point. Use `math.find` or `operation://catalog` to inspect the installed
  operation library, then use `math.run` for one bounded computation.
