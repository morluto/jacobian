# Jacobian documentation

Jacobian is a stateless mathematical tool layer for agents: `math.find`
discovers typed operations, `math.run` executes one bounded operation, and the
caller composes the returned mathematical values.

## Start here

- [Executable mathematical vocabulary](explanation/executable-mathematical-vocabulary.md) —
  why operations are semantically atomic and how vocabulary gaps are discovered.
- [Product model](explanation/product-blueprint.md) — caller/server ownership and
  public contract boundaries.
- [Architecture](explanation/architecture.md) — package structure and execution
  boundaries.
- [Discover and invoke operations](how-to/invoke-domain-operations.md) — use
  `math.find` and `math.run`.
- [Backend requirements](how-to/backend-requirements.md) — maintained Python
  backends and optional Lean.
- [Remote deployment](how-to/deploy-remote-mcp.md) — serve Jacobian over MCP.

## Reference

- [Tool surface](reference/tools.md) — exact MCP contracts.
- [Domain operation library](reference/domain-operation-library.md) — design
  rules for public mathematical operations.
- [Public operation admission](reference/public-operation-admission.md) — what
  belongs in the agent-visible catalog.
- [Native Python API](reference/python-api.md) — supported `jacobian.math`
  functions and values.
- [Operation references](reference/operations/index.md) — external-boundary
  notes that are not captured by the live schema.
- [Testing strategy](reference/testing-strategy.md) — validation ownership and
  focused test lanes.

The live `math.find` catalog is authoritative for available operations and their
current schemas.

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
contracts.
