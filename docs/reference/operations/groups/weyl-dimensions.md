# Weyl dimensions

`root_system.weyl_dimension.compute` returns the exact dimension of the
irreducible finite-dimensional representation with a supplied dominant integral
highest weight. Weight coordinates are pairings against the ordered simple
coroots, so they are fundamental-weight coordinates on the Cartan axis.

For each positive root `alpha`, the result records the positive root and its
coroot coordinates, together with the numerator and denominator
`<lambda + rho, alpha^vee>` and `<rho, alpha^vee>`. The dimension is the exact
integer product of these rational factors. A disconnected Cartan matrix is
accepted as a semisimple direct sum, and its component dimensions multiply.

The operation admits finite Cartan rank at most 8 and at most 120 positive
roots. Highest-weight coordinates are nonnegative integers of at most 64 bits;
root count, integer growth, work, and output size are bounded before root
enumeration and product construction.
