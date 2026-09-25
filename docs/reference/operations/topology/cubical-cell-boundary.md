# Oriented cubical cell boundary

`topology.cubical.cell.boundary.compute` returns the signed codimension-one
faces of one `CubicalCell`. If its nondegenerate intervals occur at ambient
axes `i_1 < ... < i_k`, then

```text
∂Q = Σ_(j=1)^k (-1)^(j-1) (Q_(i_j)^+ - Q_(i_j)^-).
```

Each term includes the face, its coefficient, the ambient axis, and the fixed
endpoint. Coefficients use the canonical orientation induced by increasing
ambient axis order. Degenerate intervals do not participate in the sign
alternation. A point has no codimension-one faces, so its boundary is empty.

The operation accepts the existing integer-lattice cell representation, with
coordinates limited to 64 decimal digits. At most 20 terms can be produced;
the predicted JSON result is admitted against a 32 KiB bound before faces are
constructed. The signed terms compose with the cubical chain complex operation.

The orientation follows the standard product orientation on elementary cubes:
the boundary of an oriented interval is its upper endpoint minus its lower
endpoint, extended across the ordered product with alternating signs.
