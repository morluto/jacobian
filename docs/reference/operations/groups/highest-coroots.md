# Highest positive coroots

`root_system.highest_coroots.compute` returns one highest positive coroot for
each connected Cartan component. In a component with ordered simple-coroot
basis `alpha_i^vee`, coroot dominance means

`alpha^vee <= beta^vee` exactly when every coefficient of
`beta^vee - alpha^vee` is nonnegative.

Thus the returned coroot is the unique maximum in the component's complete
positive-coroot poset. Reducible data have one such maximum per factor; the
operation does not compare roots from distinct factors. Each component record
contains its ordered simple-root indices, the comarks (the highest-coroot
coefficients in that local basis), the complete-rank coroot vector, the root
whose coroot it is, and the zero-based index of that source pair in
`root_system.coroots.compute`'s canonical positive-root ordering. The source
coroot table and its Cartan datum are retained unchanged.

This distinction matters in non-simply-laced types: a highest coroot need not
be the coroot of the highest root. For the Cartan matrix `[[2,-3],[-1,2]]` of
G2, the highest coroot has comarks `(2,3)`, with preimage root `(2,1)`. In
`A1 x B2`, the result has separate components and local comark tuples `(1,)`
and `(1,2)`.

The operation admits root closure, exact coroot construction, componentwise
dominance comparisons, and the complete serialized result before expanding the
positive-coroot table. It supports finite Cartan rank at most 8 and at most 120
positive roots.
