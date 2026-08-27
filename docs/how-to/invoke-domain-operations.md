# Discover and invoke domain operations

Use `math.find` progressively, then call `math.run` once with the selected
operation ID and a `payload` matching its request model. The ordinary path is:

1. Search globally for one atomic mathematical outcome when the operation is
   unknown. Use a short phrase such as "integer nth root" or "real root
   isolation", not a complete proof goal. Search ranking is deterministic
   lexical retrieval, not a recommendation.
2. Inspect the selected operation before forming an unfamiliar payload. Its
   schemas and examples are authoritative for that installed catalog.
3. Run exactly that operation with one typed payload.
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

After inspection, run the selected operation. For example:

```json
{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}
```

## Compose a returned value

The result is returned directly. When a subsequent operation accepts a canonical
value from the first result, pass that complete value unchanged into its typed
field. Do not rebuild a context-bearing value from selected fields: its
normalization, axes, ambient object, or other mathematical context may be part
of the value. Extract a scalar, witness, or projection only when the inspected
input schema explicitly asks for it.

For example, `sat.cnf.canonicalize` returns `output.cnf`, and both `sat.solve`
and `sat.assignment.check` accept that entire canonical CNF as their `cnf`
payload field. The caller chooses whether solving or checking is the useful
next move; Jacobian does not retain values, workflow state, artifacts, ports,
or workspace documents.

## Interpret outcomes before continuing

A successful tool call returns the operation's own mathematical result model.
For `sat.solve`, `SAT` and `UNSAT` have their stated meanings, while `UNKNOWN`
is a non-conclusion; its `exhausted` field may say whether a time, work, or
memory budget was exhausted. A malformed payload or unknown operation ID is a
tool error, not a mathematical result. A client timeout aborts transport and is
also not a conclusion. Preserve the selected operation, exact payload or digest,
and the changed budget, backend, representation, or partition before retrying.

## Use the same contract from the CLI

The installed CLI is useful for checking one operation locally before wiring it
into an MCP host. `inspect` prints the exact current schemas and examples;
`run` accepts the same payload as `math.run`:

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

The CLI prints a JSON envelope containing the operation ID, its typed `output`,
and `runtime_ms`. It does not retain results or create workflow state.
