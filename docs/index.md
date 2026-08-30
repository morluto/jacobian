# Jacobian documentation

Jacobian is a stateless mathematical tool layer for agents: `math.find`
discovers typed operations, `math.run` executes one bounded operation, and the
caller composes the returned mathematical values.

## Choose a path

### Use Jacobian

- [Discover and invoke operations](how-to/invoke-domain-operations.md) — the
  agent workflow: search, inspect, run, and compose one bounded result.
- [Tool surface](reference/tools.md) — exact MCP contracts and result/error
  boundaries.
- [Native Python API](reference/python-api.md) — supported `jacobian.math`
  functions and canonical values.

### Understand the model

- [Executable mathematical vocabulary](explanation/executable-mathematical-vocabulary.md) —
  why operations are semantically atomic and how vocabulary gaps are discovered.
- [Product model](explanation/product-blueprint.md) — caller/server ownership and
  public contract boundaries.
- [Architecture](explanation/architecture.md) — package structure and execution
  boundaries.

### Deploy or contribute

- [Backend requirements](how-to/backend-requirements.md) — maintained Python
  backends.
- [Remote deployment](how-to/deploy-remote-mcp.md) — serve Jacobian over MCP.
- [Domain operation library](reference/domain-operation-library.md) — design
  rules for public mathematical operations.
- [Public operation admission](reference/public-operation-admission.md) — what
  belongs in the agent-visible catalog.

## Reference

- [Mathematical backends](reference/mathematical-backends.md) — adapter,
  conversion, and external-process contracts.
- [Known backend defects](reference/backend-known-defects.md) — upstream
  defects that adapters compensate for, with guard tests.
- [Operation references](reference/operations/index.md) — external-boundary
  notes that are not captured by the live schema.
- [Testing strategy](reference/testing-strategy.md) — validation ownership and
  focused test lanes.

The live `math.find` catalog is authoritative for available operations and their
current schemas.

## Contributing

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing code or public
contracts.
