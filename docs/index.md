# Jacobian documentation

Jacobian's documentation follows the
[Diátaxis framework](https://diataxis.fr/), organized by what the reader is
trying to do. Start with a tutorial when learning the system, use a how-to
guide for a specific task, consult reference material for exact contracts, and
read the explanations for design rationale.

Jacobian exposes composable mathematical capabilities through an MCP server,
CLI, and Python library. An agent still owns the mathematical strategy:
choosing a representation, candidate, and next operation. Jacobian supplies
the executable part when exact computation, bounded search, solver evidence,
or formal proof checking is useful.

Each capability has one mathematically atomic, agent-visible outcome. A search
may return a concrete witness or counterexample; a separate authorized checker
can establish whether that exact object proves the stated claim in its declared
scope. Optional workflows preserve intermediate artifacts. This keeps
heuristic, computed, and independently verified results distinct. The
[product model](explanation/product-blueprint.md) defines the capability
contract and ownership boundaries.

The current 0.6 architecture is pre-stable. Current reference documents and the
installed catalog define the supported capabilities and interfaces.
Evaluations guide portfolio behavior and maintenance; they do not grant
verification authority.

## Project control documents

These documents track the current product contract and its ongoing evolution:

| Question | Document | Status |
| --- | --- | --- |
| What product is Jacobian building? | [Product model](explanation/product-blueprint.md) | Active product direction |
| What does the system currently look like? | [Architecture](explanation/architecture.md) | Current implementation and trust boundaries |
| What direction is the project taking? | [Product goals](explanation/goals.md) | Rolling goals pursued in parallel |
| What is installed now? | [Tool surface](reference/tools.md) and runtime `capability://catalog` | Current interface rules and installation-specific inventory |

## Tutorials

Tutorials are guided learning paths. They assume no prior Jacobian experience
and build toward a complete result.

- [Find and verify a counterexample](tutorials/first-verified-result.md) shows
  the boundary between an unverified evaluator result and independently
  verified evidence.
- [Retrieve a Lean theorem and check a proof](tutorials/lean-declaration-discovery.md)
  composes computed declaration retrieval with independent Lean replay.

## How-to guides

How-to guides assume you already understand Jacobian's basic model and need to
complete a specific task.

- [Discover, invoke, and verify domain capabilities](how-to/invoke-domain-capabilities.md)
- [Configure an agent from a source checkout](how-to/setup-agent-from-source.md)
- [Install optional backends](how-to/install-optional-backends.md)
- [Troubleshoot Z3 installation on macOS](how-to/troubleshoot-z3-macos.md)
- [Run the Codex visibility evaluation](how-to/run-codex-visibility-evaluation.md)
- [Deploy the remote MCP server](how-to/deploy-remote-mcp.md)
- [Author a Harbor benchmark task](how-to/author-harbor-benchmark-task.md)
- [Migrate the benchmark portfolio](how-to/migrate-agent-workflow-benchmark.md)
- [Run agent evaluations](how-to/run-agent-evaluations.md)

## Reference

Reference documents define exact interfaces, records, gates, and test
expectations.

**Cross-cutting references:**

- [Tool surface](reference/tools.md) — MCP resources, tools, and invocation contracts
- [Domain operation library](reference/domain-operation-library.md) — built-in producer, bounded-search, artifact, and exact-replay contracts
- [Native Python API](reference/python-api.md) — supported native-value modules
- [Provider runtime](reference/provider-runtime.md) — backend availability, compatibility, and identity
- [Plugin conformance](reference/plugin-conformance.md) — plugin contract and conformance gates
- [Persistent state format](reference/state-format.md) — supported migration floor and fresh-store transition
- [Testing strategy](reference/testing-strategy.md) — validation layers, commands, and CI responsibilities

**Domain-owned references:** [Capability references](reference/capabilities/index.md)
grouped by owning domain (graphs, matrix, polynomial, Lean, SAT/SMT, finite
math, number theory, linear algebra, topology, geometry).
Adding an operation or provider does not require editing a central list; each
domain owns its own subdirectory.

**Evaluation references:** [Benchmark contracts](reference/evaluations/index.md)
and [evaluation methods](reference/evaluations/index.md) — Harbor task
contracts, dataset inventory, validation gates, workflow observation,
performance measurement, and regression policy.

**Reference scenarios:** [Worked cases](reference/scenarios/index.md) —
mathematical scenario catalog and certified-homology case.

Use the runtime `capability://catalog` and `math.find` for the
installed capability inventory and exact operation schemas.

## Explanation

Explanation documents describe why Jacobian has its current boundaries and
how its major parts fit together.

- [Architecture](explanation/architecture.md)
- [Product model](explanation/product-blueprint.md)
- [Product goals](explanation/goals.md)
- [About the hero image](explanation/hero-image.md)
- [Durable search runtime](explanation/search-runtime.md)

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
documentation. The
[atomic capability portfolio](contributing/atomic-capability-portfolio.md)
records the formal-first backend research, ordering, installation tradeoffs,
and evaluation gates used to decide which mathematical slices to build next.

Harbor benchmark authoring and verifier work is encoded in the repository-local
[`harbor-benchmarks`](../.agents/skills/harbor-benchmarks/SKILL.md) skill. Use
the exact task gate for routine benchmark changes; control/treatment model runs
are explicit operator-run evidence exercises. For hosted operation, follow
[Deploy the remote MCP server](how-to/deploy-remote-mcp.md); ignored `tmp/`
records are evidence from one host, not maintained instructions.

When adding a document, place it according to the reader's need:

- `tutorials/` for a guided learning experience;
- `how-to/` for completing one task;
- `reference/` for contracts and lookup material;
- `explanation/` for design context and decisions; and
- `contributing/` for maintainer-facing research and planning records.

Do not mix active direction with supported behavior. Product goals guide
priorities; only an applicable specification or conformance document defines a
release contract.
