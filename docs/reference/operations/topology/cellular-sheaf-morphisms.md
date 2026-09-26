# Cellular-sheaf morphisms

The `cellular_sheaf.morphism.compute` operation checks a pointwise map between
two checked finite cellular sheaves on the same simplicial complex and exact
coefficient field. Supply one exact matrix
`phi_sigma: F(sigma) -> G(sigma)` for each simplex in canonical face order.
Rows use the target stalk basis and columns use the source stalk basis.

The operation checks every cover square

```text
rho_G(sigma < tau) phi_sigma = phi_tau rho_F(sigma < tau).
```

Because both parents are checked functors on the finite face poset, commuting
cover squares imply naturality for every comparable face inclusion. The result
retains both parent sheaves and the canonicalized component matrices. A failed
candidate is returned with `natural: false` and the first failing cover pair.

`cellular_sheaf.morphism.compose` composes two serialized results pointwise.
It re-admits and checks the naturality of both caller-supplied inputs, requires
the intermediate sheaf values to agree exactly, and returns the original
source, final target, and composed component matrices. This makes the result
usable after serialization without relying on unchecked caller claims.

The operations bound parent diagrams, pointwise component cells, exact scalar
digits, square-multiplication work, and worst-case scalar output growth before
arithmetic. The current per-coefficient input limit is 64 decimal digits.
