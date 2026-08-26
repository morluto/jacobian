# Discover and invoke domain operations

Use `math.find` progressively, then call `math.run` once with the selected
operation ID and a `payload` matching its request model. Use `search` when the
operation is unknown, `browse` for compact operation-ID-sorted pages in an
unfamiliar domain, and `inspect` for the selected operation's exact typed request,
result, and valid examples. For example:

```json
{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}
```

The result is returned directly. To continue a calculation, retain the relevant
typed fields, update the hypothesis, and pass those fields in the next operation's
payload. Jacobian does not retain caller values, workflow state, artifacts, ports,
or workspace documents.

## Use the same contract from the CLI

The installed CLI is useful for checking one operation locally before wiring it
into an MCP host. `inspect` prints the exact current schemas and examples;
`run` accepts the same payload as `math.run`:

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

The CLI prints a JSON envelope containing the operation ID, its typed `output`,
`runtime_ms`, and `queue_wait_ms` (zero for direct CLI execution). It does not
retain results or create workflow state.
