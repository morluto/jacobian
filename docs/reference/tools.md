# Tool reference

Jacobian exposes two MCP tools for atomic mathematics.

- `math.find` searches the immutable built-in operation catalog.
- `math.run` executes one operation with a typed `payload`.

`math.run` accepts no state directory, artifact input, value reference, port
binder, replay record, or generic verification plan. A result is a bounded
typed mathematical value together with its execution state. Larger workflows
remain the caller's responsibility: retain a value and choose the next
operation. Domain predicates and source checks return their own typed verdicts;
the server does not create generic verification records.

```json
{"operation_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}
```

Small results are returned inline. The sole built-in MCP resource is
`operation://catalog`, the immutable catalog view. The server registers its
typed Pydantic tools directly with the MCP Python SDK.
