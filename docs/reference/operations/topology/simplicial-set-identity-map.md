# Identity map of a finite simplicial set

`topology.simplicial_set.map.identity.compute` takes a complete
`FiniteTruncatedSimplicialSet` prefix and returns a `TruncatedSimplicialMap`
whose source and target are that same prefix. In degree `n`, its row is the
identity function on the stored simplex indices `0..|X_n|-1`.

The result composes directly with other maps on the same retained carrier. The
operation uses the carrier's existing limits: degree at most 4, at most 32
simplices in each degree, and at most 96 simplices total. Since the input has
already passed the simplicial-set contract, construction copies only these
bounded index rows and does not repeat the face or degeneracy identity checks.

For the degree-0 through degree-2 prefix of `Delta[1]`, the rows are
`((0, 1), (0, 1, 2), (0, 1, 2, 3))`.
