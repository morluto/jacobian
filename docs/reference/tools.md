# Math tool surface

[Documentation home](../index.md)

- Status: Current MCP contract
- Availability is defined by the active operator-compiled catalog

Jacobian exposes exactly two MCP tools.

| Tool | Meaning |
| --- | --- |
| `math.find` | Search available built-in operations or inspect one exact ID |
| `math.run` | Execute one selected operation and return a mathematical value or checker verdict |

The complete active inventory is available from `operation://catalog`.
Catalog membership establishes availability, not recommendation,
compatibility, or verification authority.

## `math.find`

Search uses a bounded lexical query:

```json
{"request": {"op": "search", "query": "exact matrix determinant", "limit": 5}}
```

Exact inspection uses an operation ID:

```json
{"request": {"op": "inspect", "operation_id": "matrix.determinant.compute"}}
```

The request is nested because the pinned SDK publishes and enforces that
discriminated union directly. Search-only fields cannot appear on inspection,
and inspection IDs cannot appear on search.

Search proceeds in this order:

```text
lexical retrieval
  → declared input-kind and artifact-type compatibility
```

It returns lexical relevance plus factual catalog metadata:

- applicability status and stable mismatch code;
- accepted input and artifact types; and
- produced artifact types.

Applicability uses two outcomes: `INCOMPATIBLE` and
`NEEDS_MORE_TYPED_REQUIREMENTS`. Search never reports an operation as
invocable without validating its complete request.

The current search request has no full operation payload, so it reports
`NEEDS_MORE_TYPED_REQUIREMENTS` after a compatible coarse input filter and
`INCOMPATIBLE` with `INPUT_KIND_MISMATCH` or `ARTIFACT_TYPE_MISMATCH` when a
declared filter rules an operation out. It does not infer an input type from
query wording or manufacture a strong/weak confidence label. Exact request
compatibility and preflight remain facts of the selected operation invocation.

The retrieval order is not a workflow recommendation. Search does not browse,
serve inventory for an empty query, expose projection levels, publish
`next_views`, reconstruct schemas as prose, or prescribe a next operation.

Exact inspection returns the authoritative typed request and result schemas,
effect, preflight information, declared value ports, and bounded validated
examples. Runtime and checker executable identities remain operator-owned
internal state rather than discovery metadata.

## `math.run`

Run one known operation with a payload:

```json
{
  "operation_id": "integer.compute.gcd",
  "payload": {"left": "84", "right": "30"}
}
```

Operations that declare typed ports may bind inputs by opaque request-local
reference:

```json
{
  "operation_id": "finite_field.polynomial_map.fibers.compute",
  "payload": {},
  "inputs": {
    "table": {"value_ref": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}
  }
}
```

The same carrier can avoid reserializing a producer's typed result into an
independent checker request. For example, the Smith producer exposes its result
as `output.value_refs.smith_form`, while its checker declares the `candidate`
input port:

```json
{
  "operation_id": "matrix.normal_form.smith.verify",
  "payload": {"input": {"matrix": {"entries": [["2", "4"], ["6", "8"]]}}},
  "inputs": {
    "candidate": {"value_ref": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}
  }
}
```

The reference only carries the candidate value. The separate checker still
validates the complete request and independently replays the relation before it
can create a verification record.

The runtime resolves declared inputs, assembles one request, parses it once,
runs preflight, executes one semantic operation, checks the request/result
postcondition, and then publishes the result. Unknown top-level arguments and
unknown selected-payload fields fail closed. References are opaque and scoped
to the runtime that produced them; they carry no assurance. The bounded store
retains recently used references and may evict older ones, while closing the
runtime invalidates all remaining references. The catalog and inspection result
list each operation's exact input and output ports. Because each CLI command
owns and closes one runtime, request-local references are intended for immediate
MCP-session and in-process handoffs, not handoff between separate CLI invocations.

Ordinary operations return a bounded mathematical value and execution status.
Checker operations return an accepted, rejected, or non-conclusion verdict
with exact bindings. There are no dual-mode operations.

```text
Completed[T] | NonConclusion | Failed
```

Timeout, cancellation, provider failure, resource refusal, and checker
interruption do not establish a mathematical conclusion. Their public result
contains diagnostics and retained artifact lineage, never a mathematical value
or checker verdict. A completed bounded operation may still carry `UNKNOWN` or
`INCOMPLETE` in its typed result. Exact computation alone does not grant
independent verification.

## Values and resources

Small values stay inline. A response uses a request-local `value://` reference
or durable `artifact://` resource only when identity, independent retrieval,
replay, resumability, evidence binding, or size-separated transport requires
it. Carrier changes do not alter semantic identity or assurance.

The generic public resources are:

```text
operation://catalog
artifact://...
```

Operation-specific resources may exist when an installed declaration publishes
them. Generic workflow catalogs and experiment resources are not part of the
mathematical product.

## SDK projection

The server pins `mcp==2.0.0` and `mcp-types==2.0.0`. It uses MCP SDK-derived
schemas, Pydantic output validation,
`structured_output=True`, context injection, cancellation, progress, transport,
and middleware. It returns Pydantic results directly unless a real
`ResourceLink`, custom metadata, or deliberate text projection requires an
explicit MCP result.

Because the pinned SDK does not publish strict extra-argument schemas, a narrow
boundary shim rejects unknown call arguments. It can be deleted when the SDK
publishes `additionalProperties: false` and rejects unknown arguments in
conformance tests.

`math.find` is read-only and idempotent. `math.run` is non-destructive at the
fixed MCP surface but is not globally read-only or idempotent; the selected
operation's exact effect is catalog metadata.

The fixed generic executor does not expose each selected payload as a separate
host-level tool schema. Jacobian does not add prepared handles or
direct-operation aliases as a workaround.

## CLI parity

The mathematical CLI uses the same installed declarations, preflight,
execution, and publication semantics:

```text
jacobian catalog
jacobian inspect <operation-id>
jacobian run <operation-id> --json ...
jacobian run <operation-id> --file ...
```

Handwritten CLI commands are reserved for operator administration rather than
duplicating mathematical operations.
