# Finite simplicial-set products

`topology.simplicial_set.product.compute` constructs the degreewise Cartesian
product of two finite truncated simplicial sets. Both inputs must cover the
same maximum degree `N`; in each retained degree it forms
`(X × Y)_n = X_n × Y_n` in left-major order.

The returned `simplicial_set` is an ordinary `FiniteTruncatedSimplicialSet`, so
it can be passed directly to existing simplicial-map and normalized-chain
operations. Its wire labels are canonical local IDs (`p0`, `p1`, ...). The
`pair_axes` field records the corresponding exact `(left_index, right_index)`
for every product simplex. `left_projection` and `right_projection` are typed
`TruncatedSimplicialMap` values bound to the product and their respective
factors, so they can be passed directly to simplicial-map composition.

For each face or degeneracy, the output table applies that same map to each
factor and combines the resulting indices using the target degree's right
factor size. Consequently all simplicial identities hold componentwise. The
`checked_identities` field gives the number of visible identity instances for
the returned prefix; construction uses the factor identities and componentwise
definition rather than replaying checks on computed rows.

Admission occurs before axes and map tables are expanded. The operation bounds
each product degree by 32 simplices, the complete prefix by 96 simplices, map
rows by 50,000, identity work by 100,000, and conservative serialized output
size by 1,000,000 bytes. A product exceeding a carrier or work bound is rejected
as a resource-admission error; no partial product is returned.

The product result retains both factor values, the product carrier, exact pair
axes, and both projections. The product carrier itself is directly composable
with operations that consume a finite truncated simplicial set.
