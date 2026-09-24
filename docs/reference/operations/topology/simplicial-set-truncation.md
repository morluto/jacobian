# Finite simplicial-set truncation

[Topology operations](index.md) · [Tool reference](../../tools.md)

`topology.simplicial_set.truncate.compute` retains the exact prefix in degrees
`0..N` of a finite truncated simplicial set. The request degree must be at most
the source maximum degree. The result keeps the source's degree sets and every
face and degeneracy map whose source and target lie in the retained prefix.

The operation rechecks all simplicial identities visible in the retained
prefix and returns that `FiniteTruncatedSimplicialSet` directly, with its own
identity count. It makes no claim about degrees above `N`. Reapplying
truncation at the same degree leaves the prefix unchanged.

Degree sets, map rows, and estimated output are admitted before slicing. The
current carrier limits the maximum degree to 4, each degree to 32 simplices, and
the full prefix to 96 simplices.
