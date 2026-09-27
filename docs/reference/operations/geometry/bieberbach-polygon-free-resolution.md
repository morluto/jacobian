# Bieberbach polygon free-resolution value

The native `polygon_free_resolution` helper accepts the source-bound
`BieberbachFaceOrbitComplex` value from
`crystallographic.quotient_face_orbits.compute`. It reconstructs that complex
from the retained positive fundamental-polygon result before using its
group-labelled boundary entries. The helper remains outside the public
operation catalog because its output packages data already retained by that
carrier with the trivial-module augmentation.

The value records free **right** modules over the integral group ring of the
represented crystallographic extension in degrees 0, 1, and 2. A sparse
incidence entry `(c, g)` means `c * (g · e_target)` in the boundary of
`e_source`; the deck element acts on the right of the target basis cell, so
these are free right `ZΓ`-modules. The
augmentation sends each degree-zero orbit generator to 1 in the trivial module
`Z`. Applying it to the group-ring boundaries gives the retained integral
quotient chain complex, which composes with `chain_complex.homology_groups`.

The contract is restricted to torsion-free rank-two groups and verified convex
polygons with at most 32 vertices and 32 facets. The check that the polygon is
a fundamental domain and that the action is torsion-free makes its universal
cover the contractible Euclidean plane. Its cellular chains are therefore a
free `ZGamma`-resolution; the output matrices are those cellular boundaries,
not an assertion inferred from `d^2 = 0` alone. Higher dimensions, arbitrary
polytope search, and resolutions for groups with torsion are outside this
helper's supported scope.

HAPcryst documents the same construction pattern: obtain a Bieberbach
fundamental domain, compute its face lattice and boundary, then construct a
resolution ([Chapter 4](https://gap-packages.github.io/hapcryst/doc/chap4_mj.html)).
Jacobian's helper uses its own typed exact data and does not delegate to
GAP, Polymake, or HAP.
