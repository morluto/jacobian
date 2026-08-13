# Repository Guidelines

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, documentation,
commits, and pull requests. This file lists only Jacobian-specific constraints.
Load the [product model](docs/explanation/product-blueprint.md),
[architecture](docs/explanation/architecture.md), or
[tool reference](docs/reference/tools.md) when needed. For built-in mathematical
operations, also use the
[domain operation library reference](docs/reference/domain-operation-library.md).

## Product Constraints

Jacobian is a **toolbox of atomic math tools** for agents. It is not a workflow
engine: the agent owns decomposition, sequencing, checker choice, and stopping.

| Agent verb | MCP tool | Meaning |
| --- | --- | --- |
| Search | `math.find` | Find or inspect math tools (IDs, schemas, examples). |
| Execute | `math.run` | Run **one** tool → **mathematical value** (or checker **verdict**). |
| Inventory | `capability://catalog` | Full catalog when needed. |

See [product model](docs/explanation/product-blueprint.md) and
[Search and execute](docs/explanation/architecture.md#search-and-execute).
Not a required sequence: agents may run a known ID, search first, or re-find
mid-investigation.

**Results are math-first.** Ordinary tools return calculations (GCD, matrix,
path, factors, …) plus execution status. Do not add generic assurance,
completeness, scope, or obligation knobs to ordinary results; bounded status
belongs in the domain result that defines it.

**Checker tools are additional tools.** Independent check is a **separate
catalog ID** (e.g. `….verify`, `lean.check`), not a role on the producer. **No
dual-mode tools.**

- Server: typed contracts, resource bounds, catalog install, checker
  authorization.
- Model: representation, which tools to run, how to compose values, when to
  call checker tools, when to stop.
- Discovery must not become a hidden planner (“recommended next step”).
- Evaluations reward correct math, useful intermediate values, safety, and
  efficiency—not a fixed tool-call sequence.

**Naming.** Agent-facing: **math tool** / **operation**. Code/catalog often
still say **capability** for the same thing. No parallel rename without a plan.

Tools stay atomic, searchable, and freely composable. No prescribed proof
strategy, verification order, or stopping criteria in discovery, ranking,
prompts, or adapters.

Design against the portfolio. Reuse values/artifacts; prefer composable
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
not invoke `math.run`, construct a capability runtime, or expose MCP,
artifact, provider-loading, or installation objects.

### Mathematical interoperability

Operations interoperate through shared, typed domain values and artifacts—not
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

Canonical decimal strings are wire and persistence values, not computational
values. Use the canonical conversion API before calling backends or constructing
results. Do not directly apply `int()` or `str()` to canonical components or
change `sys.set_int_max_str_digits()` as a workaround. Keep backend coercion in
thin adapters, and test above 4,300 digits whenever the contract permits it.

Keep Pydantic models authoritative at capability, persistence, artifact, and
wire boundaries. Domain implementations and operation factories must preserve
their concrete request and result types: do not accept
`Callable[[ContractModel], ContractModel]` or cast a validated request back to a
domain model. A bounded operation records exact, incomplete, or unknown status
in its domain result instead of adding generic completeness or obligation
wrappers. When a native API and a capability expose the same outcome, share one
typed mathematical kernel and use explicit domain-owned conversions rather
than duplicating the mathematics or introducing a generic conversion framework.

Shared abstractions require two surviving production paths and must delete the
older duplication in the same change. An ordinary operation should need at most
one public domain function, one request model when necessary, one rich result
model when necessary, one semantic operation declaration, and one external
publication binding only when inline transport is insufficient. Publication
owns transport only; it does not own mathematical validation, applicability,
provider selection, effects, parsing, or checker authority.

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

Construct wire envelopes only at the final capability or protocol projection.
Mathematical functions, typed operation executors, artifact services, and
checker services return their owned typed values or terminal states; they do
not construct `CapabilityResult`. Do not hide artifact writes inside an
`OperationSpec.execute` callable to satisfy this rule. When an operation needs
a domain-specific durable schema or parent closure, keep that publication in a
narrow named domain publisher and pass its typed projection to the one final
envelope constructor.

Measure expensive checker source and dependency identity at authorization and
inside the bounded checker worker, not during catalog discovery, compatibility
selection, or ordinary runtime startup. Registry reads validate persisted
identity and authority; the execution boundary remeasures executable bytes.

At the MCP boundary, prefer MCP Python SDK 2.0 high-level typed returns. Return
Pydantic result models directly and let the SDK derive the output schema,
validate results, and populate `content` and `structured_content`. Use an
explicit `CallToolResult` only when a response genuinely requires MCP content
blocks such as `ResourceLink`, custom metadata, or a deliberate text
projection. Set `structured_output=True` so unsupported return annotations fail
during registration.

Return small, bounded mathematical values inline. Materialize an artifact only
when the result needs durable identity, independent retrieval, replay,
resumability, evidence binding, or size-separated transport. Do not add
persistence flags or generic retention policy to ordinary computations.

Built-in mathematical producers belong in explicit domain bundles. Do not add
global operation registries, recursive package discovery, import-time
registration, or mechanical wrappers for backend functions. Producers remain
capped at `COMPUTED`; domain-owned checker declarations do not authorize
themselves.

`DomainBundle` is a semantic declaration, not an installation escape hatch. It
must not own installer callbacks, runtime services, storage collaborators, or
dependency-resolution policy. A capability family that genuinely needs a
special artifact/checker lifecycle is an explicitly named portfolio component
at the composition root; do not add a generic knob to every ordinary bundle for
one exceptional installer. An operation may bind a typed computational backend
that owns no runtime, storage, publication, installation, or checker authority;
that backend is part of execution, not application lifecycle.

Keep availability, recommendations, compatibility, and verification authority
separate. Experimental contracts may break between versions; compatibility
applies only to supported versions. Only an operator-authorized checker
independent of proposal, search, and evaluation may return `VERIFIED`.

Follow the
[ownership model](docs/explanation/product-blueprint.md#ownership-model).
Keep strategy out of the kernel, semantics out of generic contracts, and
checker authorization out of plugins and search code.

## Fail-Closed Verification Rules

- Treat `TIMEOUT`, `CANCELLED`, `ERROR`, incomplete enumeration, and failure to
  find a witness as non-conclusions.
- Never promote an evaluator score, solver status, model answer, or search
  result directly to `VERIFIED`.
- Keep execution status, input validity, mathematical conclusion, assurance,
  and evidence type separate.
- Bind `VERIFIED` evidence to the exact claim, semantics, candidate, scope,
  certificate format, and checker identity.
- Plugins and search code cannot authorize checkers or change trust policy.
- Independent checkers cannot depend on the search implementation they certify.

## Repository Gotchas

- Before final validation, use `make check` plus the named lane that owns the
  changed behavior on the final tree (`make check-external` for Lean/Mathlib,
  `make test-provider` for optional or maintained Python providers). In a
  shared checkout, agents must own disjoint paths and must not switch
  branches, stage, commit, clean, or rewrite shared files until their work
  is integrated.
- Jacobian is pre-stable. Current reference documents and the installed catalog
  define the supported surface; they do not order capability research.
- Validate the complete Pydantic request model before preflight, provider calls,
  computation, allocation, or artifact writes. This includes relationships among
  individually valid fields: parents, characteristics, presentations, axes,
  bases, labels, and bound identities must agree where the operation requires
  them. JSON Schema supports discovery; it does not replace cross-field model
  validation. Exercise incompatible-but-individually-valid values through the
  serialized installed-operation boundary and assert an invalid-request result
  with no execution or publication.
- Mathematical inputs are not presumed confidential. Public diagnostics should
  expose a stable domain reason, path, limit, and recovery direction—not
  arbitrary rejected values, which may be unbounded or user-controlled. This
  projection must not add another validation pass.
- A `COMPLETED` bounded operation may return a domain result marked `UNKNOWN` or
  `INCOMPLETE`. Execution completion alone does not establish optimality or a
  mathematical conclusion.
- Include every first-class artifact reference, including verification records,
  in the result's `artifact_uris`.
- An unavailable optional native or formal provider must remove only the
  affected capabilities. A missing or mismatched maintained Python backend is
  a broken installation and must fail runtime construction clearly.
- Keep `deep_review.md` local; it is ignored and is not design source material.
- Keep worked cases in reference scenarios and benchmarks.

## Agent Workflow Entry Points

Capability work remains agent-directed and is not coupled to a mandatory
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
- Optional native and formal backends are absent by default and their
  capabilities are correctly omitted: `lean.check` prints
  `lean.check is not installed` on `init`/startup
  (the pinned Lean 4.31.0 toolchain is not installed), and external solver
  executables (`cadical`, `drat-trim`, `carcara`) are not on `PATH`. This does
  not break the kernel, catalog, or the core test suites. Only install Lean/elan
  or those executables when specifically exercising `lean_runtime` tests or SAT
  proof-artifact capabilities.
- `make test-unit` is the cheap unit lane. `make quick` adds lint; `make check`
  adds lint and typecheck. `make check-all` explicitly reproduces the Lean-free
  ordinary CI matrix. Use `make test-all-ci` only
  for an explicit exhaustive local reproduction; it takes this worktree's
  exhaustive validation lease (`make validation-status`). Default `uv run pytest` does
  not collect Lean, storage, process, or MCP; use the matching `make test-*`
  target for those trees. Never run bare `uv run pytest` as a substitute for
  the complete specialist matrix.
- Only the coordinating agent may start an exhaustive test lane. Never delegate
  one to a parallel agent sharing the host. Before an exceptional broad run,
  inspect active processes for pytest jobs from this checkout and stop or wait
  for them; concurrent runtime/store/subprocess suites turn per-test timeouts
  into a host-contention detector rather than useful failure evidence.
- SQLite is one visible contention point, but not the sole cause: full-runtime
  construction also performs durable filesystem publication, subprocess
  startup, schema registration, and CPU-heavy capability setup. A timeout
  observed in `PRAGMA`, `fsync`, `os.link`, or process startup under concurrent
  suites must be reproduced with the owning focused test before it is treated
  as a product defect.
- Quick end-to-end smoke of the product surface: `uv run jacobian --state-dir .jacobian init`
  (CLI), `uv run jacobian-mcp` for one local stdio server, or
  `uv run jacobian-remote-mcp --host 127.0.0.1 --port 8000 --allow-anonymous`
  for an explicit remote test host. Remote hosting requires `--allow-anonymous`
  or `--auth-tokens-file`; those options are intentionally absent from the local
  entry point. The runnable
  `docs/tutorials/first-verified-result.md` script demonstrates one end-to-end
  investigation that includes discovery and independent verification.
