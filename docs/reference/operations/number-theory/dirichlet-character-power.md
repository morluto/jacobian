# Exact Dirichlet-character powers

`dirichlet_character.power.compute` returns \(\chi^k\) in the identical
finite unit-group parent for a bounded signed integer `k`. On a dual cyclic
axis of order `m`, the output coordinate is `k*c mod m`; thus `k=0` gives the
principal character and negative powers are exact group inverses. The input
exponent is admitted by the shared 256-digit integer bound before coordinate
scaling.

For Dirichlet characters all values on units are roots of unity, so pointwise
inverse equals complex conjugation. The power operation nevertheless retains
the group-law convention explicitly and composes with character multiplication.
