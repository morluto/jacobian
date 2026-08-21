# Diameter and radius

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.distance_matrix.compute` exposes the complete distance value for a
bounded connected graph. A disconnected input has a typed non-applicability
result; it is not silently coerced into a finite metric. Radius and diameter
remain available as the native `jacobian.math.graphs.radius` and
`jacobian.math.graphs.diameter` projections.

These operations use the submitted graph directly. Jacobian does not retain a
graph, cache an all-pairs matrix, or expose a separate verification lifecycle.
