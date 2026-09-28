# Rational SU(2) lattice gauge operations

The gauge catalog has a distinct exact SU(2) carrier built from rational unit
quaternions. `lattice_gauge.su2.gauge_transform.compute` applies
\(U'_{u\to v}=g_u U_{u\to v}g_v^{-1}\) to every oriented edge and preserves
the source lattice and frame values. `lattice_gauge.su2.holonomy.compute`
composes an oriented path in traversal order. The native helper
`jacobian.math.gauge.su2_wilson_trace` accepts a closed, source-bound
holonomy, recomputes the ordered product from that field and path, and
returns the fundamental trace \(2\operatorname{Re}(U)\) as an exact rational;
it rejects a caller-authored quaternion that differs from the recomposition.
This deterministic projection of the published holonomy is native API rather
than a catalog operation; the Wilson loop trace is discovered through
`lattice_gauge.su2.holonomy.compute`.

On the advertised norm-one domain, quaternion conjugation and group inversion
are the same map, so one catalog operation
(`quaternion.rational_unit.conjugate.compute`, with inverse discovery
vocabulary) publishes both names; the scalar-part accessor is likewise a
native helper. The multiplication and conjugation operations are
independently available under `quaternion.rational_unit.*`. These compose
with the gauge values through the same canonical quaternion representation.
Components are reduced rationals; exact unit norm is required. Coordinate
growth and multiplication work are bounded before expansion.

The exact covariance fixture uses four links and four vertex frames. It checks
each transformed link and the ordered plaquette identity
\(U'_p=g_0U_pg_0^{-1}\), then compares the exact Wilson traces. This is a
finite rational computation and makes no claim about continuum Yang–Mills
theory.
