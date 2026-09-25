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

`cellular_sheaf.morphism.cochain_map` turns a natural morphism into its
degreewise maps on cellular sheaf cochains. Each returned dense matrix is
block diagonal in the canonical simplex order: its block at a simplex is the
stalk component there, and its rows and columns are bound to the returned
target and source cochain coordinate axes. This provides the typed map needed
to compare or transport cochain-level data; it does not itself compute an
induced map on cohomology. As with composition, the consumer rechecks
naturality because a serialized `natural` flag alone is not an established
mathematical fact.

The operations bound parent diagrams, pointwise component cells, exact scalar
digits, square-multiplication work, induced matrix cells, and worst-case scalar
output growth before arithmetic. The current per-coefficient input limit is
64 decimal digits.

`cellular_sheaf.morphism.kernel.compute` computes the categorical kernel
stalkwise over `QQ` or `GF(p)`. Its result includes a based kernel sheaf, every
derived restriction map expressed in the kernel stalk bases, and the canonical
inclusion morphism into the source. The operation rechecks naturality from the
component matrices and verifies that each source restriction preserves the
pointwise kernel before it returns the induced map. A zero-dimensional kernel
stalk is represented by an empty basis and correctly shaped empty matrices.

`cellular_sheaf.morphism.image.compute` computes the pointwise image sheaf over
the same exact fields. At each simplex, the canonical pivot columns of the
stalk component give a based image stalk. Naturality implies that each target
restriction carries one image stalk into the next; the operation solves for
the induced image restriction in those bases. Its result includes the
inclusion into the target and the factor map from the source, whose composite
reconstructs the supplied morphism. Zero images retain all simplex axes with
zero-dimensional stalks and correctly shaped empty matrices. This follows the
cellular-sheaf convention as a functor from the face poset to vector spaces;
see Hansen and Ghrist, [Toward a Spectral Theory of Cellular
Sheaves](https://doi.org/10.1007/s41468-019-00038-7).

## Pointwise cokernels

`cellular_sheaf.morphism.cokernel.compute` returns the pointwise quotient
sheaf `coker(phi)` and its canonical projection from the target sheaf. At each
simplex, the quotient stalk is `G(sigma) / im(phi_sigma)`. Naturality of `phi`
ensures each target restriction carries the source image into the target image,
so it induces a well-defined map on quotient stalks. The operation binds the
quotient to the same complex and coefficient field and returns explicit based
quotient coordinates, including zero-dimensional quotients. A target-stalk
representative for each quotient basis vector is retained at every simplex;
these representatives define the stalk basis and need not assemble into a
natural map of sheaves.
