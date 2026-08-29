# Tool reference

Jacobian exposes every admitted mathematical operation as one directly callable
MCP tool. The tool name is the unchanged operation ID, its input schema is the
owner-local request contract, and its structured output is the owner-local
result contract. For example:

```text
integer.compute.extended_gcd({"left":"84","right":"30"})
```

returns the gcd and Bézout coefficients directly, without a generic
`operation_id`, `payload`, or `output` envelope.

Two fixed tools remain alongside those direct operations:

- `math.find` searches, browses, or inspects the immutable built-in operation
  catalog. Direct execution does not require it.
- `math.run` is the transitional generic execution surface retained while
  direct discovery and catalog-scale behavior are evaluated. New callers should
  use direct operation tools.

Built-in membership follows the
[public mathematical operation admission contract](public-operation-admission.md),
which keeps the public catalog distinct from the broader native Python API.

Larger workflows are caller-owned: retain the returned value and choose the
next operation. When its inspected input schema accepts a canonical value from
the first result, pass that value unchanged; otherwise construct the requested
payload from the relevant fields. Incomplete or unknown outcomes belong to the
operation's own result model.

For the ordinary discovery-to-direct-execution path, see
[Discover and invoke operations](../how-to/invoke-domain-operations.md).

## Execution deadlines

Exact mathematical operations may legitimately run for minutes or longer.
Absent an explicit latency requirement, callers should not impose one short
timeout on every direct operation call. An operation's admitted work and
declared resource budget determine its execution envelope; any MCP or client
read timeout must cover that envelope plus bounded transport overhead.

An outer timeout aborts transport and is not a mathematical result. Preserve
the operation ID and version, exact payload or digest, resource budget, elapsed
time, error, timeout layer, and repository revision before retrying or
reporting a gap. A retry should state what changed: budget, backend,
representation, or deterministic partition.

When semantic catalog discovery is useful, use `math.find` progressively:
`search` finds a few relevance-ranked candidates, `browse` pages compact
operation cards in operation-ID order (optionally within one exact primary
namespace), and `inspect` supplies the selected operation's exact input/output
schemas and valid examples. Search and browse results retain `catalog_resource`
as the explicit pointer to the bulk catalog export.

## Form a direct request from a tool contract

Client tool discovery loads the direct operation's callable schema. Start from
one of its valid examples, when `math.find` inspection publishes one, and adapt
it to the mathematical input. Otherwise form the call arguments from the input
schema and field descriptions. They state the required representation,
including units, bounds, and canonical encodings or ordering where they matter.

An `INVALID_PARAMS` response from a direct operation means either that its JSON
object was structurally malformed or that the operation's mathematical or
resource admission rejected an otherwise well-formed request. The structured
diagnostic identifies bounded field locations and codes without echoing raw
caller values. A timeout or backend failure is an operational error instead; it
does not show that the request is mathematically inadmissible and establishes no
mathematical conclusion.

`browse` is recomputed from immutable declarations on every request, with a
caller-supplied pagination cursor. The built-in MCP resource
`operation://catalog` provides an exact bulk export. Ordinary capability routing
should use client tool discovery; use `math.find` for mathematical vocabulary
search and exact catalog inspection.
