# Repository Guidelines

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, documentation,
commits, and pull requests. This file lists only Jacobian-specific constraints.
Load the [product model](docs/explanation/product-blueprint.md),
[architecture](docs/explanation/architecture.md), or
[tool reference](docs/reference/tools.md) when needed. For built-in mathematical
operations, also use the
[domain operation library reference](docs/reference/domain-operation-library.md).

## Product Constraints

Jacobian is a **toolbox of atomic math tools** for agents, not a workflow
engine and not a trust-OS with explore/verify research phases.

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
path, factors, …) plus execution status. They do not primarily return
HEURISTIC/COMPUTED/VERIFIED slogans. Optional envelope fields may exist on the
wire during migration; do not design new behavior around them as the product.

**Checker tools are additional tools.** Independent check is a **separate
catalog ID** (e.g. `….verify`, `lean.check`), not a mode on the producer. **No
dual-mode tools.** Legacy `mode` / dual-mode descriptors:
[#1143](https://github.com/morluto/jacobian/issues/1143).

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

Prefer thin adapters to maintained mathematical systems. Pin versions when
reproducibility, certificates, or verification depend on them.
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

Capabilities interoperate through shared, typed domain values and artifacts—not
backend-specific objects, JSON round-trips, or wire encodings. Reuse existing
contract models and typed kernels; add explicit domain-owned conversions when
representations differ. Cover producer-to-consumer compatibility and canonical
or backend-native round trips in tests. Architecture checks must reject internal
JSON round-trips and unsafe canonical conversions.

Canonical decimal strings are wire and persistence values, not computational
values. Use the canonical conversion API before calling backends or constructing
results. Do not directly apply `int()` or `str()` to canonical components or
change `sys.set_int_max_str_digits()` as a workaround. Keep backend coercion in
thin adapters, and test above 4,300 digits whenever the contract permits it.

Keep Pydantic models authoritative at capability, persistence, artifact, and
wire boundaries. Domain implementations and operation factories must preserve
their concrete request, result, and obligation types: do not accept
`Callable[[ContractModel], ContractModel]`, cast a validated request back to a
domain model, or erase bounded-search obligation types. When a native API and a
capability expose the same outcome, share one typed mathematical kernel and use
explicit domain-owned conversions rather than duplicating the mathematics or
introducing a generic conversion framework.

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

- Before final validation, use `make test-plan BASE=<revision>` and run the
  selected gate on the final tree. In a shared checkout, agents must own
  disjoint paths and must not switch branches, stage, commit, clean, or rewrite
  shared files until their work is integrated.
- Jacobian is pre-stable. Current reference documents and the installed catalog
  define the supported surface; they do not order capability research.
- Validate the complete Pydantic request model before computation or artifact
  writes. JSON Schema supports discovery; it does not replace cross-field
  validation.
- `COMPLETED` bounded execution may still have `UNKNOWN` completeness and open
  obligations. Execution completion does not establish optimality or a
  mathematical conclusion.
- Include every first-class artifact reference, including verification records,
  in the result's `artifact_uris`.
- An unavailable optional provider must remove only the affected capabilities;
  unrelated kernel startup and catalog entries remain available.
- Keep `deep_review.md` local; it is ignored and is not design source material.
- Keep worked cases in reference scenarios and benchmarks.

## Agent Workflow Entry Points

Capability work remains agent-directed and is not coupled to a mandatory
development workflow. For Harbor task authoring and verifier changes, use the
repository-local [`harbor-benchmarks`](.agents/skills/harbor-benchmarks/SKILL.md)
skill and its exact task validation path. Control/treatment model evaluations
are explicit operator-run evidence exercises, not routine development gates.

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
`uv sync --locked --dev`, so dependencies (including the `flint`/`smt` dev-group
backends `python-flint`, `cvc5`, `z3-solver`) are already installed when a
session starts. Standard dev, test, lint, and build commands live in the
`Makefile` (`make help`) and `CONTRIBUTING.md`; use those rather than duplicating
them.

Non-obvious caveats:

- If a fresh non-login shell can't find `uv`, run `export PATH="$HOME/.local/bin:$PATH"`.
- Optional backends are absent by default and their capabilities are correctly
  omitted: `lean.check` prints `lean.check is not installed` on `init`/startup
  (the pinned Lean 4.31.0 toolchain is not installed), and external solver
  executables (`cadical`, `drat-trim`, `carcara`) are not on `PATH`. This does
  not break the kernel, catalog, or the core test suites. Only install Lean/elan
  or those executables when specifically exercising `lean_runtime` tests or SAT
  proof-artifact capabilities.
- `make test-unit` is the quick unit lane and `make check` combines it with lint
  and typecheck. Use `make test-all-ci` only for an explicit exhaustive local
  reproduction. Never run bare `uv run pytest` across the whole suite — it mixes
  provider and Lean boundary tests into one pool; use a focused `make test-*`
  target instead.
- Only the coordinating agent may start an exhaustive test lane. Never delegate
  one to a parallel agent sharing the host. Before an exceptional broad run,
  inspect active processes for pytest jobs from this checkout and stop or wait
  for them; concurrent runtime/store/subprocess suites turn the 60-second test
  timeout into a host-contention detector rather than useful failure evidence.
- SQLite is one visible contention point, but not the sole cause: full-runtime
  construction also performs durable filesystem publication, subprocess
  startup, schema registration, and CPU-heavy capability setup. A timeout
  observed in `PRAGMA`, `fsync`, `os.link`, or process startup under concurrent
  suites must be reproduced with the owning focused test before it is treated
  as a product defect.
- Quick end-to-end smoke of the product surface: `uv run jacobian --state-dir .jacobian init`
  (CLI), or start the MCP server with
  `uv run jacobian-mcp --transport streamable-http --host 127.0.0.1 --port 8000 --allow-anonymous`
  (remote transports require `--allow-anonymous` or `--auth-tokens-file`; stdio is
  the default transport). The runnable
  `docs/tutorials/first-verified-result.md` script demonstrates one end-to-end
  investigation that includes discovery and independent verification.
