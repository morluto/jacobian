# Jacobian documentation

Jacobian's documentation follows the
[Diátaxis framework](https://diataxis.fr/), organized by what the reader is
trying to do. Start with a tutorial when learning the system, use a how-to
guide for a specific task, consult reference material for exact contracts, and
read the explanations for design rationale.

Jacobian is a **toolbox of atomic math tools** for AI agents: find them with
`math.find`, run them with `math.run`, get **mathematical results**, and
compose those values across turns. Checker tools are optional **additional**
catalog IDs—separate from producers. Catalog entries are often still called
*capabilities* in the API. The [product model](explanation/product-blueprint.md)
and [Search and execute](explanation/architecture.md#search-and-execute) define
the contract.

Jacobian is pre-stable. The product, architecture, operation-library, and tool
documents define the current contract; the installed catalog defines which
operations are available in a particular checkout. Evaluations guide portfolio
quality and do not grant formal authority.

## Project control documents

These documents define the current product contract:

| Question | Document | Status |
| --- | --- | --- |
| What is Jacobian? | [Product model](explanation/product-blueprint.md) | Product and ownership model |
| How is it structured? | [Architecture](explanation/architecture.md) | Dependencies and trust boundaries |
| What does MCP expose? | [Tool surface](reference/tools.md) | Fixed MCP projection |
| What operations are installed now? | Runtime `capability://catalog` | Current installation-specific inventory |
| What work is open? | GitHub issues (e.g. architecture epics) | Implementation priorities live in issues, not a parallel goals doc |

## Tutorials

Tutorials are guided learning paths. They assume no prior Jacobian experience
and build toward a complete result.

- [Compute and independently check a determinant](tutorials/first-verified-result.md)
  shows the boundary between an exact producer and a separate checker.
- [Retrieve a Lean theorem and check a proof](tutorials/lean-declaration-discovery.md)
  composes computed declaration retrieval with independent Lean replay.

## How-to guides

How-to guides assume you already understand Jacobian's basic model and need to
complete a specific task.

- [Discover, invoke, and check domain math tools](how-to/invoke-domain-capabilities.md)
- [Configure an agent from a source checkout](how-to/setup-agent-from-source.md)
- [Install native and formal providers](how-to/install-native-and-formal-providers.md)
- [Troubleshoot Z3 installation on macOS](how-to/troubleshoot-z3-macos.md)
- [Run the MCP visibility evaluation](how-to/run-codex-visibility-evaluation.md)
- [Deploy the remote MCP server](how-to/deploy-remote-mcp.md)
- [Author a Harbor benchmark task](how-to/author-harbor-benchmark-task.md)
- [Run agent evaluations](how-to/run-agent-evaluations.md)

## Reference

Reference documents define exact interfaces, records, gates, and test
expectations.

**Cross-cutting references:**

- [Tool surface](reference/tools.md) — MCP resources, tools, and invocation contracts
- [Domain operation library](reference/domain-operation-library.md) — built-in producer, bounded-search, artifact, and exact-replay contracts
- [Native Python API](reference/python-api.md) — supported native-value modules
- [Provider runtime](reference/provider-runtime.md) — backend availability, compatibility, and identity
- [Persistent state format](reference/state-format.md) — supported migration floor and fresh-store transition
- [Testing strategy](reference/testing-strategy.md) — validation layers, commands, and CI responsibilities

**Domain-owned references:** [Capability references](reference/capabilities/index.md)
grouped by owning domain (graphs, matrix, polynomial, Lean, SAT/SMT, finite
math, number theory, linear algebra, topology, geometry).
Adding an operation or provider does not require editing a central list; each
domain owns its own subdirectory.

**Evaluation references:** [Benchmark contracts](reference/evaluations/benchmark-contracts.md)
and [evaluation methods](reference/evaluations/evaluation-methods.md) — Harbor task
contracts, dataset inventory, validation gates, model observations,
performance measurement, and regression policy.

**Reference scenarios:** [Worked cases](reference/scenarios/index.md) —
mathematical scenario catalog and certified-homology case.

Use the runtime `capability://catalog` and `math.find` for the
installed capability inventory and exact operation schemas.

## Explanation

- [Product model](explanation/product-blueprint.md) — what the product is
- [Architecture](explanation/architecture.md) — host shape, search/execute,
  ownership and durable execution

Do not add parallel “direction”, “goals”, or portfolio-planning novels under
`explanation/`. Product intent lives in those two documents; open work lives
in GitHub issues.

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
documentation.

Harbor benchmark authoring and verifier work uses the repository-local
[`harbor-benchmarks`](../.agents/skills/harbor-benchmarks/SKILL.md) skill.
Source-grounded reliability probes based on recently resolved conjectures use
[`recent-conjecture-evaluations`](../.agents/skills/recent-conjecture-evaluations/SKILL.md).
For hosted operation, follow
[Deploy the remote MCP server](how-to/deploy-remote-mcp.md); ignored `tmp/`
records are host evidence, not source of truth.

When adding a document, place it by reader need:

- `tutorials/` — guided learning
- `how-to/` — one task
- `reference/` — contracts and lookup
- `explanation/` — only product model and architecture unless a feature needs
  a dedicated operational reference that does not fit architecture

Do not mix product intent with supported release behavior. Concrete work lives
in GitHub issues.
