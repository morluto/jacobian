# Discover and invoke domain operations

Use `math.find` progressively, then call `math.run` once with the selected
operation ID and a `payload` matching its request model. Use `search` when the
operation is unknown, `browse` for compact operation-ID-sorted pages in an
unfamiliar domain, and `inspect` for the selected operation's exact typed request,
result, and valid examples. For example:

```json
{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}
```

The result is returned directly. To continue a calculation, retain the relevant
typed fields, update the hypothesis, and pass those fields in the next operation's
payload. Jacobian does not retain caller values, workflow state, artifacts, ports,
or workspace documents.
