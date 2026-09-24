# Rational SU(2) lattice gauge operations

The gauge catalog has a distinct exact SU(2) carrier built from rational unit
quaternions. `lattice_gauge.su2.gauge_transform.compute` applies
\(U'_{u\to v}=g_u U_{u\to v}g_v^{-1}\) to every oriented edge and preserves
the source lattice and frame values. `lattice_gauge.su2.holonomy.compute`
composes an oriented path in traversal order. `lattice_gauge.su2.wilson_trace.compute`
accepts a closed, source-bound holonomy and returns the fundamental trace
\(2\operatorname{Re}(U)\) as an exact rational.

The quaternion multiplication, inverse, conjugate, and scalar-part operations
are independently available under `quaternion.rational_unit.*`. These compose
with the gauge values through the same canonical quaternion representation.
Components are reduced rationals; exact unit norm is required. Coordinate
growth and multiplication work are bounded before expansion.

The exact covariance fixture uses four links and four vertex frames. It checks
each transformed link and the ordered plaquette identity
\(U'_p=g_0U_pg_0^{-1}\), then compares the exact Wilson traces. This is a
finite rational computation and makes no claim about continuum Yang–Mills
theory.
