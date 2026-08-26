# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` searches, browses, or inspects the immutable built-in operation
  catalog.
- `math.run` executes one operation with a typed `payload` and returns that
  operation's typed mathematical result.

Built-in membership follows the
[public mathematical operation admission contract](public-operation-admission.md),
which keeps the public catalog distinct from the broader native Python API.

Larger workflows are caller-owned: retain the returned value, choose the next
operation, and construct its payload from the relevant fields. Incomplete or
unknown outcomes belong to the operation's own result model.

## Execution deadlines

Exact mathematical operations may legitimately run for minutes or longer.
Absent an explicit latency requirement, callers should not impose one short
timeout on every `math.run` call. An operation's admitted work and declared
resource budget determine its execution envelope; any MCP or client read
timeout must cover that envelope plus bounded transport overhead.

An outer timeout aborts transport and is not a mathematical result. Preserve
the operation ID and version, exact payload or digest, resource budget, elapsed
time, error, timeout layer, and repository revision before retrying or
reporting a gap. A retry should state what changed: budget, backend,
representation, or deterministic partition.

```json
{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}
```

Use `math.find` progressively: `search` finds a few relevance-ranked candidates,
`browse` pages compact operation cards in operation-ID order (optionally within a
domain), and `inspect` supplies the selected operation's exact input/output
schemas and valid examples.

## Form a payload from an inspected contract

Inspect an operation before constructing an unfamiliar payload. Start from one
of its valid examples, when it has one, and adapt it to the mathematical input.
Otherwise form the payload from the input schema and field descriptions. They
state the required representation, including units, bounds, and canonical
encodings or ordering where they matter. An error from `math.run` means the
request did not meet that operation's contract; use its diagnostic to make the
smallest correction before drawing a mathematical conclusion from any result.

`browse` is recomputed from immutable declarations on every request, with a
caller-supplied pagination cursor. The built-in MCP resource
`operation://catalog` provides an exact bulk export; ordinary discovery should
prefer `math.find`. The server registers typed Pydantic tools directly with the
MCP Python SDK.
