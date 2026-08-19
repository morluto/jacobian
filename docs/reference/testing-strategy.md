# Testing strategy

[Documentation home](../index.md)

Tests prove one observable mathematical or transport contract at a time.

## Routine validation

Run the bounded local handoff before sharing a code change:

```sh
make setup
make check
```

`make check` runs Ruff, mypy, and the Lean-free math, catalog, dispatch, CLI,
and tooling owners. Add the narrowest named lane when a change crosses its
real boundary:

| Change | Additional check |
| --- | --- |
| MCP tool schema or transport | `make test-mcp` |
| One mathematical domain | `make test-math TESTS=tests/math/logic/test_tools.py` |
| Cross-owner behavior | `make test-integration` |
| Child-process behavior | `make test-process` |
| Documentation | `make docs-linkcheck` |

`make check-all` is an intentional broad reproduction. Do not use a full suite
as a substitute for a focused regression test.

## What to test

For an operation, test the typed request boundary, the domain result, and a
real caller-visible invocation when the MCP projection changed. The integration
catalog test executes every advertised invocation example. When one result feeds
another operation, test that composition through the next typed payload.

Use property tests for canonicalization and algebraic invariants when they
state the contract more directly than examples. Use maintained libraries in
their owning domain tests rather than mocking their algorithms. A timeout,
cancellation, unavailable external executable, or solver `UNKNOWN` is never a
positive mathematical conclusion.

`lean.check` is the retained external process boundary. Its tests cover request
bounds, process cleanup, timeout/error projection, and typed diagnostics.

## Documentation acceptance

Documentation should describe current behavior rather than refactor history.
Link to the [product blueprint](../explanation/product-blueprint.md) for product
philosophy, and keep tool/reference pages focused on the contracts they own.

Run `make docs-linkcheck` after changing Markdown. It validates relative links,
documented Make commands, and documented test paths.
