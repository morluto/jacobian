# Math tool surface

[Documentation home](../index.md)

- Status: Target pre-stable MCP contract
- Installed membership is runtime-defined

Jacobian exposes exactly two MCP tools.

| Tool | Meaning |
| --- | --- |
| `math.find` | Search installed operations or inspect one exact ID |
| `math.run` | Execute one selected operation and return a mathematical value or checker verdict |

The complete installed inventory is available from `capability://catalog`.
Catalog membership establishes availability, not recommendation,
compatibility, or verification authority.

## `math.find`

Search uses a bounded lexical query:

```json
{"op": "search", "query": "exact matrix determinant", "limit": 5}
```

Exact inspection uses an operation ID:

```json
{"op": "inspect", "capability_id": "matrix.determinant.compute"}
```

The public SDK shape may nest this discriminated request under `request` if the
pinned MCP SDK cannot publish and enforce the flat union without handwritten
schema or dispatch code. The semantic operations remain `search` and
`inspect` either way.

Search proceeds in this order:

```text
lexical retrieval
  → exact typed compatibility
  → optional full-request preflight
```

It returns lexical relevance plus factual execution metadata where known:

- applicability status and stable mismatch code;
- provider availability;
- checker availability and checker scope;
- effect; and
- aggregate-cost admission status.

Applicability uses stable outcomes: `APPLICABLE`, `INCOMPATIBLE`,
`NEEDS_MORE_TYPED_REQUIREMENTS`, `PROVIDER_UNAVAILABLE`,
`CHECKER_UNAVAILABLE`, and `PORTFOLIO_GAP`. A checker that requires two inputs
is never reported as invocable from one.

The retrieval order is not a workflow recommendation. Search does not browse,
serve inventory for an empty query, expose projection levels, publish
`next_views`, reconstruct schemas as prose, or prescribe a next operation.

Exact inspection returns the authoritative typed request and result schemas,
effect, provider requirements, preflight information, declared value ports, and
bounded validated examples. Current availability remains sourced from the live
catalog rather than static documentation.

## `math.run`

Run one known operation with a payload:

```json
{
  "capability_id": "integer.compute.gcd",
  "payload": {"left": "84", "right": "30"}
}
```

After typed ports are available, callers may bind declared inputs by opaque
request-local reference:

```json
{
  "capability_id": "matrix.rank.compute",
  "payload": {},
  "inputs": {"matrix": {"value_ref": "value://opaque-id"}}
}
```

The runtime resolves declared inputs, assembles one request, parses it once,
runs preflight, executes one semantic operation, checks the request/result
postcondition, and then publishes the result. Unknown top-level arguments and
unknown selected-payload fields fail closed.

Ordinary operations return a bounded mathematical value and execution status.
Checker operations return an accepted, rejected, or non-conclusion verdict
with exact bindings. There are no dual-mode operations.

```text
Completed[T] | NonConclusion | Failed
```

Timeout, cancellation, provider failure, resource refusal, and checker
interruption do not establish a mathematical conclusion. A completed bounded
operation may still carry `UNKNOWN` or `INCOMPLETE` in its typed result. Exact
computation alone does not grant independent verification.

## Values and resources

Small values stay inline. A response uses a request-local `value://` reference
or durable `artifact://` resource only when identity, independent retrieval,
replay, resumability, evidence binding, or size-separated transport requires
it. Carrier changes do not alter semantic identity or assurance.

The generic public resources are:

```text
capability://catalog
artifact://...
```

Operation-specific resources may exist when an installed declaration publishes
them. Generic workflow catalogs and experiment resources are not part of the
mathematical product.

## SDK projection

During the migration the server pins `mcp==2.0.0` and `mcp-types==2.0.0`. It
uses MCP SDK-derived schemas, Pydantic output validation,
`structured_output=True`, context injection, cancellation, progress, transport,
and middleware. It returns Pydantic results directly unless a real
`ResourceLink`, custom metadata, or deliberate text projection requires an
explicit MCP result.

Until python-sdk issue #3067 is resolved, a narrow boundary shim rejects
unknown call arguments. It is deleted only when the pinned SDK publishes
`additionalProperties: false` and rejects unknown arguments in conformance
tests.

`math.find` is read-only and idempotent. `math.run` is non-destructive at the
fixed MCP surface but is not globally read-only or idempotent; the selected
operation's exact effect is catalog metadata.

The fixed generic executor does not expose each selected payload as a separate
host-level tool schema. That limitation is accepted while issue #1031 remains
deferred; Jacobian does not add prepared handles or direct-operation aliases as
a workaround.

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
