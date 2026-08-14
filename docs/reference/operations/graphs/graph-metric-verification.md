# Diameter and radius

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.invariant.diameter.compute` and
`graph.invariant.radius.compute` compute their respective exact metric
invariants for a bounded connected graph. A disconnected input has a typed
non-applicability result; it is not silently coerced into a finite metric.

These operations use the submitted graph directly. Jacobian does not retain a
graph, cache an all-pairs matrix, or expose a separate verification lifecycle.
