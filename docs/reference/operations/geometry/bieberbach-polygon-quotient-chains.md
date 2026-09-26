# Two-dimensional Bieberbach polygon quotient chains

`crystallographic.quotient_face_orbits.compute` accepts a positive result from
`crystallographic.extension.fundamental_domain.check`. It rechecks that result
before using its fundamental-domain claim, then derives the quotient face
orbits from the verified directed side pairings of a rational polygon.

The current envelope is dimension two, with at most 32 polygon vertices and
32 facets. In this dimension the codimension-one faces are edges; each exact
edge map therefore determines the maps of both endpoints. The result retains
the source fundamental polygon, every directed endpoint orbit map, canonical
vertex-orbit representatives, edge-orbit representatives, group-labelled
incidences, and the augmented cellular chain complex over `ZZ`. The operation
checks that the group-labelled boundary entries compose to zero and that the
ordinary quotient differentials compose to zero.

The resulting ordinary cellular chain complex composes directly with
`chain_complex.homology_groups` for exact integral homology. Included fixtures
recover the torus chains from the unit square and the Klein bottle homology
from a glide-reflection rectangle.

The group-labelled entries are reconstructed from the checked fundamental
polygon each time this operation runs. They are retained as explanatory
resolution data, but no downstream operation accepts them as authoritative
input; integral homology consumes only the augmented `ChainComplexValue`.

This operation does not return a free `ZGamma`-resolution or a contracting
homotopy. Its group-labelled entries retain the face-incidence information
needed for a later resolution value and checker; `d^2 = 0` alone is not treated
as a proof of exactness. Higher-dimensional face orbits and general polytope
resolution construction remain outside this contract.

The architecture follows the established HAPcryst computation: construct a
fundamental domain, compute face-orbit representatives and boundary data, then
form the resolution. HAP documents tensoring a free resolution with the
trivial module before homology. Jacobian implements only the verified
two-dimensional quotient-chain step here; it does not delegate to GAP,
Polymake, or HAP.

- [HAPcryst: resolutions of crystallographic groups](https://gap-packages.github.io/hapcryst/doc/chap4_mj.html)
- [HAP: resolutions and tensoring with the trivial module](https://gap-packages.github.io/hap/doc/chap2.html)
