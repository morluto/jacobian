# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` either matches one concise mathematical need against the immutable
  built-in operation catalog or returns the exact schemas and examples for one
  known operation ID.
- `math.run` executes one operation with a typed `payload` and returns that
  operation's typed mathematical result.

Built-in membership follows the
[public mathematical operation admission contract](public-operation-admission.md),
which keeps the public catalog distinct from the broader native Python API.

MCP tool results do not inherit the canonical codec's default byte ceiling.
The Python MCP SDK limits incoming Streamable HTTP request bodies, not tool
result responses. A deployment may configure a concrete response limit, but
that is an operational delivery policy owned by the adapter: it does not narrow
the mathematical domain, shared result type, or native Python API.

Larger workflows are caller-owned: retain the returned value and choose the
next operation. When its inspected input schema accepts a canonical value from
the first result, pass that value unchanged; otherwise construct the requested
payload from the relevant fields. Incomplete or unknown outcomes belong to the
operation's own result model.

For the ordinary search-to-inspection-to-execution path, see
[Discover and invoke operations](../how-to/invoke-domain-operations.md).

## Schemas and mathematical witnesses

The [SDK boundary and value contract](value-interoperability.md#mcp-python-sdk-v2)
distinguishes structured-output validation from mathematical correctness.
MCP does not require certificates. Return exact values directly; include a
source-bound witness only when it serves the operation's mathematical purpose.

## Execution deadlines

Exact mathematical operations may legitimately run for minutes or longer.
Absent an explicit latency requirement, callers should not impose one short
timeout on every `math.run` call. An operation's admitted work and
declared resource budget determine its execution envelope; any MCP or client
read timeout must cover that envelope plus bounded transport overhead.

An outer timeout aborts transport and is not a mathematical result. Preserve
the operation ID and version, exact payload or digest, resource budget, elapsed
time, error, timeout layer, and repository revision before retrying or
reporting a gap. A retry should state what changed: budget, backend,
representation, or deterministic partition.

One request deadline covers strict parsing, owner execution, result projection,
and canonical serialization. Cancellation and deadline checks also run between
those phases; expiry after mathematical computation but before delivery is an
operational failure, not a mathematical conclusion.

## Execution non-completion and recovery

MCP distinguishes an invalid request from a valid call that could not complete.
Structural or mathematical admission failures use `INVALID_PARAMS`. Timeout,
cancellation, configured worker or host capacity exhaustion, backend failure,
and delivery failure return an agent-visible tool error (`is_error=true`). The
error is not a mathematical result and must not be interpreted as `False`,
`UNSAT`, absence of a witness, or completeness of a partial search.
This follows MCP's tool-execution error channel: the call returns an error
result that the model can inspect and respond to, rather than a protocol-level
claim that its parameters were invalid.

An agent can retry with a smaller request, a more compact representation,
another backend, or a deployment with more capacity. Diagnostics may name the
exhausted boundary, but exact results are never truncated.

Use `math.find` with `request.op="match"` and a short description of the local
result needed. Its compact matches retain `catalog_resource` as an explicit
pointer to the bulk catalog export. Then call `math.find` with
`request.op="inspect"` to obtain the selected operation's exact input/output
schemas and valid examples.

## Form a payload from an inspected contract

Inspect an operation before constructing an unfamiliar payload. Start from one
of its valid examples, when it has one, and adapt it to the mathematical input.
Otherwise form the payload from the input schema and field descriptions. They
state the required representation, including units, bounds, and canonical
encodings or ordering where they matter. An `INVALID_PARAMS` response from
`math.run` means either that the payload was structurally malformed or that the
operation's mathematical admission rejected an otherwise well-formed request.
Use its structured diagnostic to make the smallest correction before drawing a
mathematical conclusion. A timeout, cancellation, resource exhaustion, or
backend failure is an operational error instead; it does not show that the
request is mathematically inadmissible and establishes no mathematical
conclusion.

The built-in MCP resource `operation://catalog` provides an exact bulk export;
ordinary discovery should prefer `math.find`.
