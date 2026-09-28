# Translation torus quotient chains

`crystallographic.translation_torus.quotient_chains.compute` returns the
integral product cellular chain complex of a torus from a checked fundamental
parallelepiped for a pure translation group.

The accepted ranks are one through four, matching the exact finite-lattice
extension input. A source must have `2^n` vertices, `2n` facets, and one inverse
translation pairing for each pair of opposite facets. The operation checks the
source fundamental-domain result again, recovers the edge vectors from all
vertex subset sums, checks that those vectors are integral in the retained
lattice axes, and checks the opposite side translations. It retains the
canonical edge vectors as the oriented circle directions.

The quotient is the product of `n` circles. Give each circle its CW structure
with one zero-cell and one one-cell. Both boundary maps are zero, and the
product differential is the tensor differential, so every cellular boundary
is zero. There are `binomial(n, k)` product cells in degree `k`. Thus the output
is the exact based complex with these binomial basis sizes and zero integer
matrices. This agrees with the standard cellular computation of torus
homology; see [Hatcher, *Algebraic Topology*, Chapter 2](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).

The result composes with `chain_complex.homology_groups`. It does not construct
face orbits for nontrivial holonomy, a general three- or four-dimensional
Bieberbach quotient, or a free `ZGamma`-resolution.
