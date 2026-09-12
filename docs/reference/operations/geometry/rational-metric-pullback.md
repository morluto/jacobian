# Rational metric pullback

[Geometry operation references](index.md) · [Operation references](../index.md)

The `differential_geometry.rational_metric.pullback.compute` operation returns
the exact covariant tensor

\[
  (F^*g)_{ab} = \sum_{i,j}(\partial_aF^i)\,g_{ij}(F(x))\,(\partial_bF^j).
\]

The metric and map must use the same ordered target axis. The result uses the
map's ordered source axis and retains the metric and map as typed sources. It
does not claim that the map is injective or immersive, and a rank-deficient
map may therefore produce a degenerate tensor.

`pullback_locus_guard` is the complete source-side nonvanishing locus retained
from the map denominators, substituted metric denominators, inherited metric
chart guards, and the substituted metric determinant. Normalization may cancel
factors in a returned component without enlarging this construction locus.

Native callers use `jacobian.math.geometry.differential.pullback.pullback_metric`;
MCP callers discover and run the operation by its ID. Both paths share the
same exact rational DAG admission, bounded worker, and canonical tensor result.
