# Exact rational metric curvature profiles

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`differential_geometry.rational_metric.curvature_profile.compute` computes the
complete exact curvature profile of one symmetric rational coordinate metric.
The native entry point is `jacobian.math.geometry.differential.metrics.curvature_profile`;
the catalog request wraps the same `RationalCoordinateMetric` value.

The metric is a covariant rank-two tensor on one ordered coordinate axis
`(x_1, ..., x_n)`, with `1 <= n <= 4`, and uses the explicit
`GENERIC_NONDEGENERATE_LOCUS` chart semantics. The operation establishes only
the formal pseudo-Riemannian coordinate calculation: it does not assert a
signature, positivity, completeness, global manifold, or coordinate domain.

## Exact result

The returned `RationalMetricCurvatureProfile` retains the source metric,
coordinate axis, tensor variance, and nonzero-denominator guards. It contains:

- the inverse metric `g^ij`;
- the Levi--Civita connection `Gamma^k_ij` (symmetric in `i,j`);
- the Riemann tensor `R^l_kij`;
- the Ricci tensor `Ric_kj`; and
- the scalar curvature `R` as a rank-zero coordinate tensor.

Components are complete dense arrays in lexicographic index order, with the
last index varying fastest. The convention is

```text
(nabla_i nabla_j - nabla_j nabla_i) v^l = R^l_kij v^k
Ric_kj = sum_i R^i_kij
R = sum_(k,j) g^kj Ric_kj
```

Every field is an exact canonical rational function over the source axis.
Zero curvature is retained as an exact zero; it does not erase the source or
nondegenerate-locus guards. The represented locus is the source denominator
locus intersected with `det(g) != 0`, together with any complete component
denominator guards required by the canonical returned fields.

## Admission and failures

Admission is semantic and occurs before symbolic expansion. It accounts for
source support and coefficient height, determinant/cofactor growth,
derivatives, rational products, shared expression-DAG nodes, complete tensor
output, retained locus guards, and mathematical coordinate allocation. The
first envelope admits at most 4 coordinates, 16,384 DAG nodes, 50,000,000
symbolic work units, 256 terms per canonical polynomial, 128 coefficient
digits, and 768 retained locus guards. Transport byte limits are not used as
mathematical admission quantities.

The metric must be symmetric and its determinant must be a nonzero rational
function. A singular metric, excessive exact work or growth, noncanonical
source fraction, deadline, cancellation, backend failure, or malformed worker
result is reported as its own failure; none is interpreted as a flat or empty
curvature profile. SymPy is used only by the private bounded exact adapter;
backend expressions never cross the native or MCP boundary.

For a minimal valid request, use the catalog example `flat_polar_chart` in
`math.find` and adapt its ordered `r, theta` axis and canonical sparse
polynomial components. The polar metric `diag(1, r^2)` has a nonzero
connection and zero Riemann tensor on `r != 0`, making it a useful composition
and locus regression.
