# Bieberbach polygon free resolution

`crystallographic.bieberbach.polygon_free_resolution.compute` accepts the
source-bound face-orbit complex from
`crystallographic.quotient_face_orbits.compute`. It reconstructs that complex
from the retained positive fundamental-polygon result before using its
group-labelled boundary entries.

The output records free modules over the integral group ring of the represented
crystallographic extension in degrees 0, 1, and 2. Each sparse incidence term
has an integer coefficient and an extension element `(lattice_translation,
holonomy_element)`. The augmentation sends each degree-zero orbit generator
to 1 in the trivial module `Z`. Applying the trivial-module augmentation to the
group-ring boundaries gives the retained integral quotient chain complex,
which composes with `chain_complex.homology_groups`.

The contract is restricted to torsion-free rank-two groups and verified convex
polygons with at most 32 vertices and 32 facets. The check that the polygon is
a fundamental domain and that the action is torsion-free makes its universal
cover the contractible Euclidean plane. Its cellular chains are therefore a
free `ZGamma`-resolution; the output matrices are those cellular boundaries,
not an assertion inferred from `d^2 = 0` alone. Higher dimensions, arbitrary
polytope search, and resolutions for groups with torsion are outside this
operation.

HAPcryst documents the same construction pattern: obtain a Bieberbach
fundamental domain, compute its face lattice and boundary, then construct a
resolution ([Chapter 4](https://gap-packages.github.io/hapcryst/doc/chap4_mj.html)).
Jacobian's operation uses its own typed exact data and does not delegate to
GAP, Polymake, or HAP.
