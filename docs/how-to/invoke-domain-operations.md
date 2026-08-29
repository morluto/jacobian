# Discover and invoke domain operations

Let the MCP client load the matching typed operation and call it directly. The
ordinary path is:

1. Search the deferred MCP surface for one atomic mathematical outcome and load
   the matching direct operation schema.
2. Form arguments from that schema and its canonical example, then call the
   operation by its exact ID.
3. Retain the mathematical result and decide the next move yourself.

Every direct tool description repeats its owner-maintained canonical argument
example because some deferred-search clients omit JSON Schema annotations from
model context. Adapt that complete example rather than guessing a nested
representation. The `operation://catalog` resource is the exact bulk-export
fallback when a catalog-wide snapshot is genuinely needed.

The operation itself accepts its owner request fields; there is no generic
operation-ID/payload wrapper. For example, call
`integer.compute.extended_gcd` with:

```json
{"left":"84","right":"30"}
```

## Compose a returned value

The result is returned directly. When a subsequent operation accepts a canonical
value from the first result, pass that complete value unchanged into its typed
field. Do not rebuild a context-bearing value from selected fields: its
normalization, axes, ambient object, or other mathematical context may be part
of the value. Extract a scalar, witness, or projection only when the loaded
input schema explicitly asks for it.

For example, `sat.cnf.canonicalize` returns `cnf`, and both `sat.solve`
and `sat.assignment.check` accept that entire canonical CNF as their `cnf`
argument. The caller chooses whether solving or checking is the useful
next move; Jacobian does not retain values, workflow state, artifacts, ports,
or workspace documents.

## Interpret outcomes before continuing

A successful direct tool call returns the operation's own mathematical result
model. For `sat.solve`, `SAT` and `UNSAT` have their stated meanings, while `UNKNOWN`
is a non-conclusion; its `exhausted` field may say whether a time, work, or
memory budget was exhausted. Malformed arguments or an unknown tool name
produce a tool error, not a mathematical result. A client timeout aborts transport and
is also not a conclusion. Preserve the selected operation, exact arguments or
digest, and the changed budget, backend, representation, or partition before
retrying.

## Use the same contract from the CLI

The installed CLI is useful for checking one operation locally before wiring it
into an MCP host. `inspect` prints the exact current schemas and examples;
`run` accepts the same owner argument object as the direct MCP tool:

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

The CLI prints a JSON envelope containing the operation ID, its typed `output`,
and `runtime_ms`. It does not retain results or create workflow state.
