# Finite-category nerve prefixes

[Topology operations](index.md) · [Tool surface](../../tools.md)

`category.finite.nerve_prefix.compute` constructs the nerve of a finite category
through a bounded degree. A degree-`n` simplex is a composable string of `n`
morphisms; degree zero contains one vertex for each object. The result retains
the category, a `FiniteTruncatedSimplicialSet` carrying the face and degeneracy
maps, and aligned morphism and vertex data for each simplex.

The face maps delete the first or last arrow at the endpoints and compose
adjacent arrows for inner faces. Degeneracies insert the identity at the
repeated vertex. Thus the returned simplicial set uses the convention that
`d_0` selects the target of a 1-simplex and `d_1` its source. The existing
simplicial-set constructor checks the simplicial identities before the result
is returned.

The request accepts a nonempty finite category and `max_degree` from zero
through the simplicial-set degree bound. The operation checks the category laws
and counts paths by endpoint before constructing simplex tables, and admits
per-degree and total simplex counts and identity-replay work against explicit
limits. Serialized result size is bounded by the transport layer rather than by
nerve admission. It returns a finite truncation only; it does not represent an
unbounded nerve.
