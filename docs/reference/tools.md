# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` searches, browses, or inspects the immutable built-in operation
  catalog.
- `math.run` executes one operation with a typed `payload` and returns that
  operation's typed mathematical result.

Built-in membership follows the
[public mathematical operation admission contract](public-operation-admission.md),
which keeps the public catalog distinct from the broader native Python API.

Larger workflows are caller-owned: retain the returned value and choose the
next operation. When its inspected input schema accepts a canonical value from
the first result, pass that value unchanged; otherwise construct the requested
payload from the relevant fields. Incomplete or unknown outcomes belong to the
operation's own result model.

For the ordinary search-to-inspection-to-execution path, see
[Discover and invoke operations](../how-to/invoke-domain-operations.md).

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

Use `math.find` progressively: `search` finds a few relevance-ranked candidates,
`browse` pages compact operation cards in operation-ID order (optionally within
one exact primary namespace), and `inspect` supplies the selected operation's
exact input/output schemas and valid examples. Search and browse results retain
`catalog_resource` as the explicit pointer to the bulk catalog export.

## Form a payload from an inspected contract

Inspect an operation before constructing an unfamiliar payload. Start from one
of its valid examples, when it has one, and adapt it to the mathematical input.
Otherwise form the payload from the input schema and field descriptions. They
state the required representation, including units, bounds, and canonical
encodings or ordering where they matter. An `INVALID_PARAMS` response from
`math.run` means either that the payload was structurally malformed or that the
operation's mathematical or resource admission rejected an otherwise
well-formed request. Use its structured diagnostic to make the smallest
correction before drawing a mathematical conclusion. A timeout or backend
failure is an operational error instead; it does not show that the request is
mathematically inadmissible and establishes no mathematical conclusion.

`browse` is recomputed from immutable declarations on every request, with a
caller-supplied pagination cursor. The built-in MCP resource
`operation://catalog` provides an exact bulk export; ordinary discovery should
prefer `math.find`.
