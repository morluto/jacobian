# Positive roots and coroots

`root_system.coroots.compute` returns each positive root together with its
coordinates in the matching simple-coroot basis and its exact squared length.
The result carries a canonical finite Cartan datum, so the ordered root and
coroot axes and their normalization remain explicit.

The operation supports crystallographic finite Cartan matrices of rank at
most 8. Positive-root closure is bounded by 120 roots. For a symmetrizer
`D = diag(d_i)`, the bilinear matrix is `D A`. If a root has simple-root
coordinates `c`, its squared length is `cᵀ D A c`, and the coefficient of
the `i`th simple coroot in the coroot is

```text
2 d_i c_i / (cᵀ D A c).
```

The exact rational coefficients are required to be nonnegative integers; a
violation is an internal contract failure. The symmetrizer normalization is
the canonical one in the returned datum. Rescaling every `d_i` in one
irreducible component changes displayed root lengths but not coroot
coordinates.

For type `B₂` with Cartan matrix `[[2,-2],[-1,2]]`, the simple root squared
lengths are 2 and 4. The positive root `(1,1)` has squared length 2 and
coroot coordinates `(1,2)` in the simple-coroot basis.

`root_system.root_to_coroot.compute` applies this transform to one supplied
positive root and returns a `RootCorootPair` bound to the canonical Cartan
datum. The root coordinates must match that datum's ordered simple-root axis
and name an actual positive root; arbitrary nonnegative vectors are rejected.
The pair carrier encodes positive roots, so negative-root conversion is outside
this operation's contract. Input coordinates, complete root closure, and
result framing are bounded before closure enumeration.

[Operation references](../index.md) · [Tool surface](../../tools.md)
