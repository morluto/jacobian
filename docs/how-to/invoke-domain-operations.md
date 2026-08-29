# Discover and invoke domain operations

Use client tool discovery to load a relevant direct Jacobian operation, then
call that operation with an object matching its request model. The ordinary
path is:

1. Let client tool search discover one atomic mathematical outcome such as
   "integer nth root" or "real root isolation", not a complete proof goal.
2. Load the selected direct operation's callable schema. Use `math.find` only
   when semantic vocabulary search, namespace browsing, or exact example
   inspection adds value beyond client tool discovery.
3. Call exactly that operation with one typed JSON object.
4. Retain the mathematical result and decide the next move yourself.

For example, search for a small number of matrix operations, then inspect an
exact candidate:

```json
{"request":{"op":"search","query":"matrix determinant","namespace":"matrix","limit":3}}
```

```json
{"request":{"op":"inspect","operation_id":"matrix.determinant.compute"}}
```

Use `browse` to map a known primary namespace in operation-ID order. A
namespace matches only the first segment of an operation ID; tags do not filter
the result set. Search and browse responses retain `catalog_resource` as the
explicit fallback for a full catalog export:

```json
{"request":{"op":"browse","namespace":"matrix","limit":20}}
```

After discovery, call the selected operation directly. For example, invoke
`integer.compute.extended_gcd` with:

```json
{"left":"84","right":"30"}
```

The result is the operation's canonical result object; it has no generic
`operation_id`, `runtime_ms`, or `output` wrapper.

## Compose a returned value

The result is returned directly. When a subsequent operation accepts a canonical
value from the first result, pass that complete value unchanged into its typed
field. Do not rebuild a context-bearing value from selected fields: its
normalization, axes, ambient object, or other mathematical context may be part
of the value. Extract a scalar, witness, or projection only when the inspected
input schema explicitly asks for it.

For example, `sat.cnf.canonicalize` returns `cnf`, and both `sat.solve`
and `sat.assignment.check` accept that entire canonical CNF as their `cnf`
request field. The caller chooses whether solving or checking is the useful
next move; Jacobian does not retain values, workflow state, artifacts, ports,
or workspace documents.

## Interpret outcomes before continuing

A successful tool call returns the operation's own mathematical result model.
For `sat.solve`, `SAT` and `UNSAT` have their stated meanings, while `UNKNOWN`
is a non-conclusion; its `exhausted` field may say whether a time, work, or
memory budget was exhausted. A malformed request or unknown tool name is a tool
error, not a mathematical result. A client timeout aborts transport and is also
not a conclusion. Preserve the selected operation, exact request or digest, and
the changed budget, backend, representation, or partition before retrying.

## Use the same contract from the CLI

The installed CLI is useful for checking one operation locally before wiring it
into an MCP host. `inspect` prints the exact current schemas and examples;
`run` accepts the same JSON object as the direct MCP tool:

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

The CLI prints a JSON envelope containing the operation ID, its typed `output`,
and `runtime_ms`. It does not retain results or create workflow state.
