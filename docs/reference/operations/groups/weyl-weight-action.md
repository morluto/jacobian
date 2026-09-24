# Weyl action on weight-lattice vectors

`weyl_group.element.act_on_weight.compute` applies a canonical finite Weyl
element to a `WeightLatticeVector` in the same ordered Cartan datum. It returns
the resulting `WeightLatticeVector` directly, so the image can be supplied to
another weight action without converting coordinates or reconstructing its
parent.

Weights use fundamental-weight coordinates. If the Cartan matrix is `A`, its
columns express simple roots in the fundamental-weight basis. For root action
matrix `M`, the induced weight action is

```text
A M A^-1.
```

The result is integral for a Weyl element and preserves the exact weight
lattice. The operation re-admits the caller-supplied root action and weight
datum, checks their common ordered Cartan parent, bounds the exact matrix work,
and admits worst-case output coordinate growth before constructing the result.
The group element must be a Weyl element, not a diagram automorphism that merely
preserves the root set.

For `A2`, the simple reflection `s0` sends the first fundamental weight
`(1, 0)` to `(-1, 1)`, since `alpha0 = (2, -1)` in fundamental-weight
coordinates. For `B2`, with Cartan matrix `[[2, -2], [-1, 2]]`, `s1` sends
`(0, 1)` to `(2, -1)`, retaining the unequal root-length convention.

This operation is distinct from `weyl_group.weight.orbit.compute`: it applies
one supplied element and returns one typed value; the orbit operation returns
the complete finite orbit under its own output bound.

[Operation references](../index.md) · [Tool surface](../../tools.md)
