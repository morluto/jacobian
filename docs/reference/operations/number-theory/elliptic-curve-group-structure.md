# Finite-field elliptic-curve group structure

`elliptic_curve.finite_field.group_structure.compute` returns the invariant
factor presentation of (E(\mathbb F_q)) and one exact point generator for
each nontrivial factor. The result uses the shared `AbelianPresentation`
carrier; each generator retains the identical finite-field curve value.

The operation exhaustively enumerates the admitted point set once and derives
the group order from that set. It finds the exponent by reducing the group
order against every point. For an elliptic curve over a finite field the
finite point group has at most two invariant factors, so its order and
exponent determine the factors. In rank two, the selected generators have
trivial cyclic-subgroup intersection; their product therefore has the full
group order.

Hasse's integer bound must be at most 4,096, and a conservative estimate for
point-order reductions and generator search must fit the work budget before
point enumeration begins. The operation returns the full group structure or
rejects admission; a partial generator search is not a group result.

[Finite-field elliptic point orders](elliptic-curve-point-order.md) ·
[Number theory operations](index.md) · [Operation reference](../index.md)
