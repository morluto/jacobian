# Discover and invoke domain operations

Use `math.find` to inspect an operation's typed request and result, then call
`math.run` once with that operation ID and a `payload` matching its request
model. For example:

```json
{"operation_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}
```

The result is returned inline. To continue a calculation, pass the relevant
typed value in the next operation's payload. Jacobian does not retain caller
values, workflow state, artifacts, ports, or workspace documents.
