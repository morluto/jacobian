# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` either matches one concise mathematical need against the immutable
  built-in operation catalog or returns the exact schemas and examples for one
  known operation ID.
- `math.run` executes one operation with a typed `payload` and returns that
  operation's typed mathematical result.

These tools adapt the Jacobian library; they do not define a second mathematical
or execution API. The library owns admission, deadlines, cancellation,
deterministic work, worker containment, backend-failure classification, and
result validation. The MCP adapter binds host request state to that library
execution envelope, projects successful values, connects optional progress to
the protocol, and converts typed library exceptions into sanitized tool errors.
MCP task identity, polling, TTL, authorization, and retained-result delivery are
serving-runtime concerns and never fields in an operation's mathematical result.

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

Distinguish protocol validity, operation admission, and execution capacity.
For `math.run`, structural or mathematical admission failures use the model-visible
tool error channel with a bounded diagnostic. The adapter raises SDK `ToolError`;
the SDK encodes a result with wire field `isError: true` (Python attribute `is_error`).
Jacobian does not map ordinary operation-payload or domain rejections to
JSON-RPC `INVALID_PARAMS`. Malformed protocol messages belong to the SDK's
protocol error path, not the mathematical operation.

Timeout, cancellation, configured worker or host capacity exhaustion, and
backend failure also use tool errors, with distinct diagnostics. A delivery
failure may prevent any error from reaching the caller. None of these failures
is a mathematical result or may be interpreted as `False`, `UNSAT`, absence of
a witness, or completeness of a partial search. A successful mathematical
`false` remains a successful tool result when it belongs to the operation's
codomain.

An agent can retry with a smaller request, a more compact representation,
another backend, or a deployment with more capacity. Diagnostics may name the
exhausted boundary, but exact results are never truncated.

Use `math.find` with `query` and a short description of the local result needed.
Its compact matches retain `catalog_resource` as an explicit pointer to the bulk
catalog export. Then call `math.find` with `operation_id` to obtain the selected
operation's exact input/output schemas and valid examples. Provide exactly one of
`query` and `operation_id`; `namespace`, `limit`, `cursor`, and `search_mode` are
search-only.

Search defaults to `search_mode: "precise"` for applicability filtering; use
`"broad"` for lexical recall. When continuing with `next_cursor`, keep `query`,
`namespace`, and `search_mode` unchanged. Only `limit` may change. An
`INVALID_CURSOR` response can be recovered by restoring those original search
settings, or by restarting the new search mode without a cursor. Cursors remain
opaque; do not edit or construct them.

An unknown operation ID has two explicit failure signals, according to the tool:

- `math.find` inspection returns its declared `kind: "error"` response with
  `error.code: "UNKNOWN_OPERATION"`; the MCP envelope remains `isError: false`.
  Clients must check `kind` as well as the MCP error flag. `INVALID_CURSOR` uses
  this same discovery error branch.
- `math.run` raises SDK `ToolError`, producing `isError: true` with the diagnostic
  rendered as JSON in the SDK's error text and no structured mathematical result.

Both unknown-ID diagnostics carry the same `code: "UNKNOWN_OPERATION"`,
`stage: "operation_resolution"`, `message`, and recovery `hint`. Neither is a
mathematical result. Recover by searching with `math.find` and `query`, then
inspecting the exact installed ID before executing it. A search with
`kind: "matches"`, an empty `matches` list, and `total_matches: 0` is successful
discovery with no candidates; it does not establish mathematical impossibility.

## Form a payload from an inspected contract

Inspect an operation before constructing an unfamiliar payload. Start from one
of its valid examples, when it has one, and adapt it to the mathematical input.
Otherwise form the payload from the input schema and field descriptions. They
state the required representation, including units, bounds, and canonical
encodings or ordering where they matter. An invalid-request tool error from
`math.run` means either that the payload was structurally malformed or that the
operation's mathematical admission rejected an otherwise well-formed request.
Use its structured diagnostic to make the smallest correction before drawing a
mathematical conclusion. A timeout, cancellation, resource exhaustion, or
backend failure is an operational error instead; it does not show that the
request is mathematically inadmissible and establishes no mathematical
conclusion.

The built-in MCP resource `operation://catalog` provides an exact bulk export;
ordinary discovery should prefer `math.find`.

## Retired operation identifiers

A retired operation identifier is a deliberate breaking change: `math.run`
reports it as unknown rather than executing a compatibility alias. Replace
`hypergraph.coloring.non_monochromatic.decide` with
`hypergraph.nonmonochromatic_vertex_coloring.q_decide`. Map its old boolean
`colorable` field to the replacement's `outcome`: `COLORABLE` or
`NOT_COLORABLE`. For `COLORABLE`, zip the old positional `coloring` tuple with
the retained `hypergraph.vertices` order to build `witness.assignments` as
`(vertex_id, color)` pairs. For example, vertices `("a", "b")` and coloring
`(1, 0)` become assignments `(("a", 1), ("b", 0))`. Preserve the hypergraph and
palette size. For `NOT_COLORABLE`, omit `witness`.

Replace `polynomial.rational.compute.evaluate` with `polynomial.map.evaluate`.
The canonical `QQ` polynomial is unchanged. Wrap the former scalar `point` in
`{"variables": ["x"], "values": [point]}`, using the polynomial's actual declared
variable name in place of `x`. The result is `{"value": ...}`; it no longer echoes
the point. Ordered axes remain explicit for renamed variables and zero
polynomials. The unified operation preserves degree-127 univariate evaluation
and the existing two-to-eight-variable regime (256 terms, total degree 64),
subject to its source/coefficient and exact-result bounds. The native
`rational_polynomial_evaluate` helper remains available without a second
catalog declaration.

## Optional backend availability

Operation declarations, matches, and browse cards include `runtime_requirements`.
These list optional system runtimes and remain independent of the server's
installation. `math.find` inspection adds `backend_availability`: bounded, current
diagnostics for those runtimes in the server environment. Inspection does not
execute the mathematical operation or guarantee its completion.

A missing or unsupported runtime needed during execution produces a tool error
with `code: BACKEND_UNAVAILABLE`, `stage: backend_execution`, `operation_id`,
`backend`, `required_version`, and an actionable `hint`. It is neither an invalid
parameter error nor a mathematical conclusion. See
[backend requirements](../how-to/backend-requirements.md).
