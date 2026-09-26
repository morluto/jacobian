# Exact global sections of a cellular sheaf

[Topology operations](index.md) · [Tool surface](../../tools.md)

`cellular_sheaf.sections.compute` returns the global sections of a checked
finite cellular sheaf as a reusable, source-bound vector space. Its coefficient
field, ambient coordinates, and stalk axes come from the input sheaf.
Scalar entries retain the coefficient field's native exact representation:
reduced `CanonicalRational` values over `QQ` and strict integer residues over
`GF(p)`. Numeric strings are not accepted as scalar inputs.

The ambient vector has one coordinate for every basis vector in every stalk.
For every strict face inclusion `sigma < tau`, the operation adds one equation
per target-stalk coordinate:

```text
rho_(sigma,tau) x_sigma - x_tau = 0
```

The returned compatibility matrix includes all comparable pairs, including
derived restrictions. Its row axes identify the face inclusion and target
basis label; its columns follow the canonical stalk and basis order. The
returned basis spans the exact matrix kernel, and each stalk evaluation matrix
projects section coordinates into that stalk. This follows the standard
cellular-sheaf definition of a global section as a compatible choice in every
stalk; the restriction direction used here is face to coface, matching the
repository’s checked sheaf diagram convention. See [Curry’s cellular-sheaf
thesis](https://arxiv.org/abs/1303.3255) and [Ghrist–Hansen,
*Toward a spectral theory of cellular sheaves*](https://doi.org/10.1007/s41468-019-00038-7).

Admission occurs before constructing the dense matrix or running exact
elimination. The operation bounds the number of simplices and stalk
coordinates, restriction input size, full compatibility-matrix cells,
nullspace work, determinant-based rational height, and exact output size.
Requests beyond these envelopes receive an operational resource error; that
does not assert that the section space is empty.

`cellular_sheaf.sections.restrict` computes the induced map on these section
spaces for an included finite subcomplex. Its result retains the source and
target sheaf, both exact section-space bases, and the matrix from source
section coordinates to target section coordinates. Inclusion is checked by
the subcomplex restriction operation; the stalk and field axes therefore
remain those of the original diagram. The aggregate serialized result is
bounded at 34,000,000 characters, and the induced matrix is bounded by 200,000
entries.
