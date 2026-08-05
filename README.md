<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="An archival-style black-and-white photograph of a mathematician working at a chalkboard, with a constant Jacobian determinant and three distinct inputs mapping to one output.">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>Executable mathematics for agents. Evidence an independent checker can replay.</strong>
</p>

<p align="center">
  An MCP server, CLI, and Python library for conjectures, counterexamples,
  exact computation, and formal proof.
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT license"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-verification-works">Verification</a> ·
  <a href="#capabilities">Capabilities</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#contributing">Contributing</a>
</p>

Jacobian gives AI agents small, composable mathematical operations rather than
one opaque solver. An agent can construct an object, compute an invariant,
search for a witness, and submit exact evidence to a separate checker. Every
step remains visible as a typed result or artifact.

A search result, solver status, model answer, timeout, or score is never
promoted directly to `VERIFIED`. Only an operator-authorized checker may emit a
verified record, bound to the exact claim, candidate, scope, semantics,
certificate format, and checker identity.

## Quickstart

The npm launcher installs Jacobian and configures supported MCP clients. For a
one-off setup without a global install, run:

```sh
npx jacobian setup
```

For repeated use, install the launcher persistently and use its commands:

```sh
npm install -g jacobian
jacobian setup
jacobian upgrade
jacobian doctor
```

For the Python distribution, install the stable package directly with:

```sh
python -m pip install jacobian
```

The launcher supports Claude, Codex, Cursor, Gemini, and OpenCode. It requires
Node.js 18 or newer, Python 3.12, and
[`uv`](https://docs.astral.sh/uv/). Run `jacobian mcp` to start the server
directly.

<details>
<summary><strong>Install from source</strong></summary>

```sh
git clone https://github.com/morluto/jacobian.git
cd jacobian
./scripts/setup-agent --client codex --profile full-python --yes
```

This performs a locked full-Python sync and configures the selected agent to
start MCP from the absolute source and state paths with `--no-sync`. It also
records a doctor report containing the Git revision, package version, catalog
digest, and provider availability. See
[Configure an agent from a source checkout](docs/how-to/setup-agent-from-source.md)
for the `core`, `full-python`, `lean`, and `external-proof` profiles, dry-run,
repeatability, and rollback behavior.

Use `uv run jacobian --help` to inspect the CLI or `uv run jacobian-mcp` to
start the MCP adapter.

</details>

## How verification works

Jacobian separates finding evidence from deciding what that evidence proves.
Suppose an agent is testing the claim **“`F` is injective.”**

<p align="center">
  <img src="docs/assets/verification-flow.jpg" width="100%" alt="The claim that F is injective leads to a candidate collision, an exact independent check, and a verification record. Missing witnesses, timeouts, cancellation, and errors remain unknown.">
</p>

**Claim → candidate witness → independent check → verification record**

| Stage | Output | What it establishes |
| --- | --- | --- |
| Claim | `F` is injective | The statement to investigate; not yet trusted |
| Search | A candidate witness `(F, p, q)` | Inspectable evidence, not a conclusion |
| Independent check | Confirm `p ≠ q` and `F(p) − F(q) = 0` exactly | The candidate is a genuine collision |
| Record | Bind the checked collision to the original claim and checker identity | The injectivity claim is `FALSE · VERIFIED` |

> **No witness is not proof.** A failed search, timeout, cancellation, or error
> leaves the claim `UNKNOWN`.

In the introductory tutorial, the same boundary appears as:

```text
evaluate.batch   →  FALSE  · HEURISTIC
witness.find     →  exact witness artifact
witness.verify   →  FALSE  · VERIFIED
```

`FALSE · HEURISTIC` is an evaluation. `FALSE · VERIFIED` is a conclusion
backed by independently checked evidence. Follow
[Find and verify a counterexample](docs/tutorials/first-verified-result.md)
for a runnable example.

## Capabilities

Capabilities are discovered at runtime through `capability://catalog`,
described with `capability.describe`, and executed with
`capability.invoke`. The installed catalog is the source of truth because
availability can depend on local backends.

| Domain | Agent-visible outcomes |
| --- | --- |
| Polynomial maps | Evaluate maps, compute Jacobians, search for collisions, independently verify collisions |
| Polynomial algebra | Normalize typed expressions, factor univariate polynomials, verify identities, verify exact system solutions |
| Exact linear algebra | Compute determinants, rank, kernels, and integer row Hermite normal forms; find and independently verify rational solutions or inconsistency certificates for `Ax = b` |
| Graphs | Construct and inspect graphs, enumerate paths, realize degree sequences, test isomorphism, search colorings |
| SAT and SMT | Find models or proof artifacts; independently replay assignments, DRAT proofs, and Alethe proofs |
| Universal algebra | Evaluate finite magma laws and search for countermodels |
| Polytopes | Compute convex combinations and linear separations |
| Lean | Discover declarations, retrieve premises, inspect proof states, and check proofs in pinned environments |

See the [tool reference](docs/reference/tools.md) for the public surface and
the [atomic capability portfolio](docs/contributing/atomic-capability-portfolio.md)
for portfolio design and evaluation gates.

## Design

Jacobian keeps four responsibilities separate:

- **Agents own strategy.** The kernel supplies mathematical operations, not a
  prescribed research workflow.
- **Capabilities expose one coherent outcome.** Useful intermediate objects,
  failures, and proof obligations remain visible.
- **Values compose directly.** Small, bounded mathematical results stay inline;
  artifacts carry reusable objects, replayable evidence, and large payloads.
- **Checkers own trust.** Plugins and search code cannot authorize a checker or
  change verification policy.

The public MCP surface stays small: the capability catalog plus the two
capability entry points, `capability.describe` and `capability.invoke`.

## Documentation

| Start here | When you need detail |
| --- | --- |
| [Documentation home](docs/index.md) | Tutorials, how-to guides, reference, and explanation |
| [Architecture](docs/explanation/architecture.md) | System shape and the independent verification boundary |
| [Product model](docs/explanation/product-blueprint.md) | Capability contracts, ownership, artifacts, and assurance |
| [Product goals](docs/explanation/goals.md) | Active priorities and research direction |
| [Tool surface](docs/reference/tools.md) | MCP resources, tools, and invocation contracts |
| [Domain operation library](docs/reference/domain-operation-library.md) | Built-in producer, bounded-search, artifact, and exact-replay contracts |
| [Provider runtime](docs/reference/provider-runtime.md) | Backend availability, compatibility, and identity |
| [Testing strategy](docs/reference/testing-strategy.md) | Validation layers, commands, and CI responsibilities |

Specialized contracts cover
[SAT artifacts](docs/reference/capabilities/sat-smt/sat-artifacts.md),
[SMT/Alethe artifacts](docs/reference/capabilities/sat-smt/smt-artifacts.md),
[exact rational linear-system evidence](docs/reference/capabilities/linear-algebra/linear-rational-solutions.md),
[exact rational matrix determinants](docs/reference/capabilities/matrix/matrix-rational-determinant.md),
[integer matrix HNF](docs/reference/capabilities/matrix/matrix-hermite-normal-form.md), and
[Lean declaration discovery](docs/reference/capabilities/lean/lean-declaration-discovery.md).
The [domain-capability how-to](docs/how-to/invoke-domain-capabilities.md)
demonstrates discovery, computed invocation, bounded-result interpretation,
and exact replay. The
[Lean formal-intermediates reference](docs/reference/capabilities/lean/lean-formal-intermediates.md)
covers proof states, premise retrieval, dependency graphs, and checked edits.

## MCP clients and deployment

`jacobian setup` registers the local server with one or more supported clients.
`jacobian upgrade` refreshes the pinned Python kernel in the launcher's managed
environment; use `npm install -g jacobian@latest` to upgrade the npm launcher
itself.
For a clone, `jacobian setup --source <checkout> --state-dir <path> --profile
full-python` explicitly binds the client to that source environment;
the maintained `scripts/setup-agent` wrapper performs the required locked sync
and doctor checks first.
The server advertises only the capability entry points;
`capability.describe(query=...)` searches compact installed outcomes before an
agent inspects an exact contract and invokes it. This is a toolbox interface:
agents own mathematical decomposition, exploration, and composition.

Clients with MCP resource support can read `jacobian://instructions` for the
operating guide and `capability://catalog` for the complete machine inventory.
Clients with prompt support can optionally request `jacobian-discover` or
`jacobian-check-evidence` for protocol scaffolding.

Remote clients can connect through Streamable HTTP or SSE with bearer-token
authentication and subject-bound tenant state. See
[Deploy the remote MCP server](docs/how-to/deploy-remote-mcp.md). Static tokens
are intended for controlled deployments, not as a hosted identity system.

From a clean clone on a systemd host, the maintained installer can deploy a
localhost endpoint, a Caddy-managed public domain, or Tailscale Funnel:

```sh
sudo ./deploy/install.sh
sudo ./deploy/install.sh --mode domain --domain math.example.org
sudo ./deploy/install.sh --mode tailscale
```

Run `./deploy/install.sh --help` or add `--dry-run` to inspect the plan first.
The public modes require a reviewed Caddy installation; Funnel additionally
requires a connected Tailscale installation. Authentication is enabled by
default, and a newly generated bearer token is printed once.

## Optional backends

Some capabilities use backends that are not installed by default:

- CaDiCaL finds SAT models and UNSAT proof artifacts.
- cvc5 produces SMT UNSAT proofs; Carcara independently checks Alethe.
- The `flint` extra provides Python-FLINT/Arb operations for exact rational
  systems, integer matrices and lattices, polynomials, and validated numerical
  computation. Individual capabilities and independent replay support depend
  on the installed catalog.
- Pinned Lean `CORE` and `MATHLIB` environments check formal certificates.

Backend availability is not verification authority. Provider output remains
unverified until the appropriate independent checker accepts its bound witness
or certificate.

<details>
<summary><strong>Lean certificates</strong></summary>

The `lean.check` capability binds an exact proposition and proof body to its
result. The bundled environments pin Lean, imports, and their allowed trust
bases; model-supplied imports and packages are rejected.

Prepare the pinned runtime with:

```sh
elan toolchain install leanprover/lean4:v4.31.0
cd lean
lake update
lake build
```

Proof-state interaction and premise retrieval are exploration aids. Their
output cannot become `VERIFIED` without a successful `lean.check`. See the
[guided declaration-discovery tutorial](docs/tutorials/lean-declaration-discovery.md).

</details>

<details>
<summary><strong>macOS and Z3</strong></summary>

The locked environment uses `z3-solver` 5.0.0.0. Its upstream macOS wheels
target macOS 13 or newer on Apple silicon and Intel. On an older release, `uv`
falls back to a source build that requires CMake, `make`, and a C++20 compiler.

Install the Xcode Command Line Tools and CMake before retrying `uv sync --dev`.
These commands report the relevant environment without changing it:

```sh
sw_vers -productVersion
uname -m
xcode-select -p
clang++ --version
cmake --version
make --version
```

See the
[`z3-solver` 5.0.0.0 files on PyPI](https://pypi.org/project/z3-solver/5.0.0.0/#files)
for the upstream wheel tags.

</details>

## Status

Jacobian 0.6.0 is a pre-stable release. Its published package, capability, and
artifact contracts describe the current supported surface; ongoing capability
research may change experimental contracts between releases.

The Python distribution contains the mathematical kernel, CLI, and MCP server.
The npm package is a thin launcher and MCP client installer for that same
implementation; it is not a separate JavaScript API.

<details>
<summary><strong>About the hero image</strong></summary>

The visual motif comes from the three-dimensional counterexample to the
Jacobian conjecture: an exact constant Jacobian determinant alongside three
distinct rational inputs with the same output. Surprising candidates are
valuable, but exact computation and independent checking establish what can be
trusted.

Terence Tao gives an
[accessible mathematical account](https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/).
The determinant identity and collision have also been
[independently formalized in Isabelle/HOL](https://isa-afp.org/entries/Jacobian_Counterexample.html).
The two-dimensional conjecture remains open.

</details>

<details>
<summary><strong>Project boundaries</strong></summary>

Jacobian does not aim to put a universal mathematical ontology, a
natural-language-to-formal-mathematics translator, distributed search
infrastructure, or an opaque generic solver into the kernel. It does not
reimplement theorem provers or SAT/MIP solvers, accept arbitrary
model-supplied executable bundles, or treat floating-point scores, timeouts,
and solver labels as proofs.

</details>

## Contributing

Jacobian uses Python 3.12, `uv`, and a small `Makefile`:

```sh
make setup
make test-unit
make check
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code. It documents
focused test commands, verification rules, documentation placement, and
pull-request expectations.

## License

[MIT](LICENSE)
