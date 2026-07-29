# Jacobian documentation

Jacobian's documentation is organized by what the reader is trying to do.
Start with a tutorial when learning the system, use a how-to guide for a
specific task, consult reference material for exact contracts, and read the
explanations for design rationale.

Jacobian exposes composable mathematical capabilities through an MCP server,
CLI, and Python library. Capabilities have mathematically atomic,
agent-visible outcomes; agents compose them into research strategies. Optional
workflows preserve intermediate artifacts, and only operator-authorized
independent checkers may promote exact evidence to a verified result. The
[product model](explanation/product-blueprint.md) defines the capability
contract and ownership boundaries.

The API and artifact formats are pre-stable. Experimental and
version-breaking adapters may be exposed before held-out evaluations show
lift. Evaluations guide portfolio behavior and maintenance; they do not grant
verification authority. Release specifications describe supported snapshots,
not a required order of development.

## Project control documents

These documents track the design while the public API and artifact formats are
still pre-stable:

| Question | Document | Status |
| --- | --- | --- |
| What product is Jacobian building? | [Product model](explanation/product-blueprint.md) | Active product direction |
| What does the system currently look like? | [Architecture](explanation/architecture.md) | Current implementation and trust boundaries |
| What direction is the project taking? | [Product goals](explanation/goals.md) | Rolling goals pursued in parallel |
| Why were cross-cutting choices made? | [Architecture decision log](explanation/adr/index.md) | Accepted decisions with release scope |
| What is the last frozen release contract? | [v0.2 specification](reference/specifications/v0.2.md) and [conformance gate](reference/conformance-v0.2.md) | Normative snapshot for `0.2.0a0`; later pre-stable releases extend it |
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
- [Deploy the remote MCP server](how-to/deploy-remote-mcp.md)

## Reference

Reference documents define exact interfaces, records, gates, and test
expectations.

- [Tool surface](reference/tools.md)
- [Domain operation library](reference/domain-operation-library.md)
- [Finite probability operations](reference/finite-probability-operations.md)
- [Exact planar geometry](reference/exact-planar-geometry.md)
- [Finite simplicial topology](reference/finite-simplicial-topology.md)
- [Finite posets](reference/finite-posets.md)
- [Recurrences and rational generating series](reference/recurrences-and-generating-series.md)
- [Provider runtime contract](reference/provider-runtime.md)
- [Lean declaration discovery](reference/lean-declaration-discovery.md)
- [Lean formal intermediates](reference/lean-formal-intermediates.md)
- [SAT artifact contracts](reference/sat-artifacts.md)
- [SMT Alethe artifact contracts](reference/smt-artifacts.md)
- [Exact rational linear-system evidence](reference/linear-rational-solutions.md)
- [Exact rational matrix determinants](reference/matrix-rational-determinant.md)
- [Graph counterexample shrinking](reference/graph-counterexample-shrinking.md)
- [Fixed-registry graph invariant batches](reference/graph-invariant-batch.md)
- [Maximum-matching certificate and verification](reference/graph-maximum-matching.md)
- [Graph diameter and radius verification](reference/graph-metric-verification.md)
- [Graph distance matrix](reference/graph-distance-matrix.md)
- [Integer prime-factorization verification](reference/integer-prime-factorization-verification.md)
- [Powerful-number decision](reference/integer-powerful-number-decision.md)
- [Bounded finite exactly-once coverage](reference/finite-coverage-verification.md)
- [Integer matrix Hermite normal form](reference/matrix-hermite-normal-form.md)
- [Typed polynomial expression normalization](reference/polynomial-expression-normalization.md)
- [Polynomial-map inverse verification](reference/polynomial-map-inverse-verification.md)
- [Lean statement proposal and elaboration](reference/lean-statement-elaboration.md)
- [Replayable Lean proof-state transitions](reference/lean-replayable-proof-states.md)
- [v0.2 specification](reference/specifications/v0.2.md)
- [v0.2 conformance specification](reference/conformance-v0.2.md)
- [Plugin conformance contract](reference/plugin-conformance.md)
- [Mathematical scenario catalog](reference/math-scenarios.md)
- [Reference benchmarks](reference/benchmarks.md)
- [Performance benchmark protocol](reference/performance-benchmarks.md)
- [Testing strategy](reference/testing-strategy.md)
- [Agent evaluation protocol](reference/agent-evaluations.md)
- [Capability workflow evaluation plan](reference/capability-workflow-evaluations.md)
- [Capability development handoffs](reference/capability-development-handoffs.md)

## Explanation

Explanation documents describe why Jacobian has its current boundaries and
how its major parts fit together.

- [Architecture](explanation/architecture.md)
- [Product model](explanation/product-blueprint.md)
- [Product goals](explanation/goals.md)
- [Durable search runtime](explanation/search-runtime.md)
- [Architecture decision log](explanation/adr/index.md)
- [ADR 0001: Python-first control plane](explanation/adr/0001-python-first-control-plane.md)
- [ADR 0002: Sealed plugin packages](explanation/adr/0002-sealed-plugin-packages.md)
- [ADR 0003: Durable search invocations](explanation/adr/0003-durable-search-invocations.md)
- [ADR 0004: Verified parameter regions](explanation/adr/0004-verified-parameter-regions.md)
- [ADR 0005: Direct epistemic workspaces](explanation/adr/0005-direct-epistemic-workspaces.md)

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
documentation. The [issue index](contributing/issues.md) records implementation
work that has been identified but not necessarily scheduled. The
[atomic capability portfolio](contributing/atomic-capability-portfolio.md)
records the formal-first backend research, ordering, installation tradeoffs,
and evaluation gates used to decide which mathematical slices to build next.
The [test-suite cost audit](contributing/test-suite-cost-audit.md) records the
measured test lanes, retained trust-boundary costs, and fast-feedback policy.

Recurring agent work is encoded in the repository-local skills under
`.agents/skills/`. Start with `develop-math-capabilities` for a complete
challenge-to-evaluation loop, or select its discovery, producer, checker, or
evaluation phase skill directly. Use the
[capability development handoff](reference/capability-development-handoffs.md)
between phases. For hosted operation, follow
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
