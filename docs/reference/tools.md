# Tool reference

Jacobian exposes every admitted operation as a direct MCP tool named by its
operation ID. Each tool publishes that owner's strict input schema, one
canonical example, and typed result schema; a successful call returns the
operation's mathematical result directly. There is no generic execution
dispatcher or operation-ID/payload wrapper.

The canonical argument example is present both as an input-schema annotation
and in the direct tool description. This keeps one owner-maintained value usable
by deferred-search clients that do not expose every JSON Schema annotation to
the model; it is not a second request contract.

Built-in membership follows the
[public mathematical operation admission contract](public-operation-admission.md),
which keeps the public catalog distinct from the broader native Python API.

Larger workflows are caller-owned: retain the returned value and choose the
next operation. When its loaded input schema accepts a canonical value from the
first result, pass that value unchanged; otherwise construct the requested
arguments from the relevant fields. Incomplete or unknown outcomes belong to
the operation's own result model.

For the ordinary load-to-execution path, see
[Discover and invoke operations](../how-to/invoke-domain-operations.md).

## Execution deadlines

Exact mathematical operations may legitimately run for minutes or longer.
Absent an explicit latency requirement, callers should not impose one short
timeout on every direct operation call. An operation's admitted work and
declared resource budget determine its execution envelope; any MCP or client
read timeout must cover that envelope plus bounded transport overhead.

An outer timeout aborts transport and is not a mathematical result. Preserve
the operation ID and version, exact arguments or digest, resource budget,
elapsed time, error, timeout layer, and repository revision before retrying or
reporting a gap. A retry should state what changed: budget, backend,
representation, or deterministic partition.

## Form direct operation arguments

Start from the canonical example on the loaded direct tool schema and adapt it
to the mathematical input. The schemas and field descriptions state the
required representation, including units, bounds, and canonical encodings or
ordering where they matter. An `INVALID_PARAMS` response from a direct
operation means either that its arguments were
structurally malformed or that the operation's mathematical or resource
admission rejected an otherwise well-formed request. Use its structured
diagnostic to make the smallest correction before drawing a mathematical
conclusion. A timeout or backend failure is an operational error instead; it
does not show that the request is mathematically inadmissible and establishes
no mathematical conclusion.

The built-in MCP resource `operation://catalog` provides an exact bulk export;
ordinary discovery should prefer deferred client tool search.
