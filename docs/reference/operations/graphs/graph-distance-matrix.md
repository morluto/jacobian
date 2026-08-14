# Graph metric operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Jacobian exposes bounded metric invariants of a typed finite graph directly.
`graph.invariant.diameter.compute` returns the exact diameter of a connected
graph, and `graph.invariant.radius.compute` returns its minimum eccentricity.
For a disconnected graph, each operation returns its typed applicability
outcome rather than a made-up numeric value.

There is no standalone distance-matrix operation, graph handle, or stored graph
value. Pass the graph itself to each operation that needs it.
