# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` searches, browses, or inspects the immutable built-in operation
  catalog.
- `math.run` executes one operation with a typed `payload`.

`math.run` accepts no state directory, artifact input, value reference, port
binder, replay record, or generic verification plan. Its small final envelope
names the operation, version, runtime, and typed `output`; the output is the
bounded mathematical value. Any incomplete or unknown outcome belongs to that
operation's own result model. Larger workflows remain the caller's
responsibility: retain a value and choose the next operation. Domain predicates
and source checks return their own typed verdicts; the server does not create
generic verification records.

```json
{"operation_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}
```

Use `math.find` progressively: `search` finds a few relevance-ranked candidates,
`browse` pages compact operation cards in operation-ID order (optionally within a
domain), and `inspect` supplies the selected operation's exact input/output
schemas and valid examples. Reuse relevant fields from each direct `math.run`
value in the next payload; the agent owns the evolving conjecture and hypothesis.

## Form a payload from an inspected contract

Inspect an operation before constructing an unfamiliar payload. Start from one
of its valid examples and adapt it to the mathematical input. Field descriptions
and the input schema state the required representation, including units, bounds,
and canonical encodings or ordering where they matter. An error from `math.run`
means the request did not meet that operation's contract; use its diagnostic to
make the smallest correction before drawing a mathematical
conclusion from any result.

`browse` is recomputed from immutable declarations on every request. Its cursor is
only caller-supplied pagination state, so it creates no catalog identity, saved
session, artifact, value reference, or server-side record. The sole built-in MCP
resource, `operation://catalog`, remains an exact bulk export rather than the
ordinary agent discovery path. The server registers typed Pydantic tools directly
with the MCP Python SDK.
