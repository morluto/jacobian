# Graph metric operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Jacobian exposes the complete bounded distance matrix of a typed finite graph.
`graph.distance_matrix.compute` returns every exact unweighted shortest-path
distance between ordered vertex pairs in canonical lexicographic vertex order.

Rows are labelled: every row names its source vertex, and its cells are the
distances from that source to the result `vertices` in their declared order.
Because each row carries its own label, the dense positional matrix stays
bound to the authoritative vertex axis and cannot be silently presented under
a different ordering — numeric-looking labels such as `"2"` and `"10"` remain
lexicographic and unambiguous. Unreachable pairs use `null`; a disconnected
input is a typed result with `connected` false rather than a failure.

Radius and diameter are cheap projections of the complete matrix, so they are
retained as `jacobian.math.graphs.radius` and
`jacobian.math.graphs.diameter`. Pass the graph itself to the public operation;
there is no graph handle or stored graph value.
