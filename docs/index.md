# Jacobian documentation

Jacobian's documentation follows the
[Diátaxis framework](https://diataxis.fr/), organized by what the reader is
trying to do. Start with a tutorial when learning the system, use a how-to
guide for a specific task, consult reference material for exact contracts, and
read the explanations for design rationale.

Jacobian is a **toolbox of atomic math tools** for AI agents: find them with
`math.find`, run them with `math.run`, get **mathematical results**, and
compose those values across turns. Domain predicates and source checks are
ordinary operations with their own typed outputs; there is no generic checker
or verification-record product. Catalog entries are often still called
*operations* in the API. The [product model](explanation/product-blueprint.md)
and [architecture](explanation/architecture.md) define the contract.

Jacobian is pre-stable. The product, architecture, operation-library, and tool
documents define the current contract; the immutable catalog exposed by a
server defines its available operations. Evaluations guide portfolio quality
and do not grant formal authority.

## Project control documents

These documents define the current product contract:

| Question | Document | Status |
| --- | --- | --- |
| What is Jacobian? | [Product model](explanation/product-blueprint.md) | Product and ownership model |
| How is it structured? | [Architecture](explanation/architecture.md) | Dependencies and trust boundaries |
| What does MCP expose? | [Tool surface](reference/tools.md) | Fixed MCP projection |
| What operations are available? | Runtime `operation://catalog` | Current immutable server inventory |
| What work is open? | GitHub issues (e.g. architecture epics) | Implementation priorities live in issues, not a parallel goals doc |

## How-to guides

How-to guides assume you already understand Jacobian's basic model and need to
complete a specific task.

- [Discover and invoke domain math tools](how-to/invoke-domain-operations.md)
- [Install native providers](how-to/install-native-and-formal-providers.md)
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
- [Domain operation library](reference/domain-operation-library.md) — built-in direct-operation contracts
- [Native Python API](reference/python-api.md) — supported native-value modules
- [Testing strategy](reference/testing-strategy.md) — validation layers, commands, and CI responsibilities

**Operation references:** [Operation references](reference/operations/index.md)
explain the two retained external boundaries. The live catalog remains the
authoritative reference for the rest of the operation library.

**Evaluation references:** [Benchmark contracts](reference/evaluations/benchmark-contracts.md)
and [evaluation methods](reference/evaluations/evaluation-methods.md) describe
the repository's separate benchmark work. They do not define the server's
operation contract.

**Reference scenarios:** [Worked cases](reference/scenarios/index.md) preserve
small mathematical workloads for documentation and testing; they are not
runtime workflows.

Use the runtime `operation://catalog` and `math.find` for the immutable
operation inventory and exact operation schemas.

## Explanation

- [Product model](explanation/product-blueprint.md) — what the product is
- [Architecture](explanation/architecture.md) — host shape and direct execution

Do not add parallel “direction”, “goals”, or portfolio-planning novels under
`explanation/`. Product intent lives in those two documents; open work lives
in GitHub issues.

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
documentation.

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
