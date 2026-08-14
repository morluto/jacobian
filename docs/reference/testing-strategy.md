# Testing strategy

[Documentation home](../index.md)

Tests prove one observable mathematical or transport contract at a time.
They do not reconstruct a complete runtime, database, publication pipeline,
tenant, or checker service.

## Routine validation

Run the bounded local handoff before sharing a code change:

```sh
make setup
make check
```

`make check` runs Ruff, mypy, and the unit suite. Add the narrowest named lane
when a change crosses its real boundary:

| Change | Additional check |
| --- | --- |
| MCP tool schema or transport | `make test-mcp` |
| One mathematical domain | `make test-unit TESTS=tests/unit/domains/test_logic_operations.py` |
| Documentation | `make docs-linkcheck` |

`make check-all` is an intentional broad reproduction. Do not use a full suite
as a substitute for a focused regression test.

## What to test

For an operation, test the typed request boundary, the domain result, and a
real caller-visible invocation when the MCP projection changed. A result that
feeds another operation should be passed as the next typed payload; do not add
tests for producer IDs, stored references, serialization round trips inside the
kernel, or hidden runtime state.

Use property tests for canonicalization and algebraic invariants when they
state the contract more directly than examples. Use maintained libraries in
their owning domain tests rather than mocking their algorithms. A timeout,
cancellation, unavailable external executable, or solver `UNKNOWN` is never a
positive mathematical conclusion.

`lean.check` is the one retained external process boundary. Its tests cover
request bounds, timeout/error projection, and typed diagnostics. They do not
create a session, cache, proof-state resource, or replay record.

## Documentation acceptance

Documentation must describe the stateless two-tool surface consistently:

- `math.find` discovers or inspects installed operation declarations;
- `math.run` executes one typed payload and returns one bounded result;
- callers keep any value needed for a later operation; and
- no page teaches SQLite state, artifact publication, value references,
  verification records, a workspace, or a migration workflow.

Run `make docs-linkcheck` after changing Markdown. It validates relative links,
documented Make commands, and documented test paths.
