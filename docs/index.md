# Jacobian documentation

Jacobian provides a native mathematical library and a catalog of published
operations. Through MCP, `math.find` matches operations or reads one exact
contract, `math.run` executes it, and the caller composes the returned values.
Native Python callers can use domain functions without a catalog or server.

## Choose a path

### Use Jacobian

- [Discover and invoke operations](how-to/invoke-domain-operations.md) — the
  agent workflow: find, inspect, run, and compose one bounded result.
- [Tool surface](reference/tools.md) — exact MCP contracts and result/error
  boundaries.
- [Native Python API](reference/python-api.md) — supported `jacobian.math`
  functions and canonical values.

- [Reduce ordered-simplex LPs](how-to/reduce-ordered-simplex-lps.md) — chain
  differences before exact basis-work admission.

### Understand the model

- [Executable mathematical vocabulary](explanation/executable-mathematical-vocabulary.md) —
  why operations are semantically atomic and how vocabulary gaps are discovered.
- [Product model](explanation/product-blueprint.md) — caller/server ownership and
  public contract boundaries.
- [Architecture](explanation/architecture.md) — the responsibility map for
  library, publication, and delivery; values versus requests; trust and limit
  ownership; package organization; and a worked operation path.

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
- [Schemas and value interoperability](reference/value-interoperability.md) —
  canonical ownership, explicit conversions, and serialized trust boundaries.
- [Graph deck values](reference/graph-decks.md) — source-bound deletion families
  versus anonymous graph-card multisets and their canonical form.
- [Rational Bernstein coordinates](reference/polynomial-bernstein.md) — exact
  polynomial basis conversion on boxes, admission bounds, and evidence.
- [Testing strategy](reference/testing-strategy.md) — validation ownership and
  focused test lanes.

The live `math.find` tool is authoritative for available operations and their
current schemas.

## Contributing

Use the relevant section of [CONTRIBUTING.md](../CONTRIBUTING.md) for validation,
documentation changes, releases, or pull requests.
