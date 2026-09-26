# Finite-field elliptic quadratic-twist relation

`elliptic_curve.finite_field.quadratic_twist_relation.compute` returns the
canonical nontrivial quadratic twist together with the exact source model and
the chosen parameter `d`. For a source curve
`y^2 = x^3 + A*x + B`, its result contains the model
`y^2 = x^3 + d^2*A*x + d^3*B`, where `d` is the first nonsquare in the
declared field's canonical coordinate order. The result is bound to the exact
same finite-field presentation at both ends.

This value preserves the mathematical relation that the existing
`quadratic_twist.compute` operation intentionally projects away when returning
only a curve. Consumers can pass `twisted_curve` directly to point-count,
Frobenius, and extension-count operations, while retaining `source_curve` and
`parameter` for comparisons. The operation does not itself compute point
counts. For the independently checkable relation between counts, if the
source trace is `t`, the twist trace is `-t` and the two cardinalities sum to
`2*(q+1)`.

The coefficient formula and trace reversal follow from the quadratic-character
point-count formula; see MIT 18.783, *Elliptic Curves*, Lecture 8, §8.5
([PDF](https://math.mit.edu/classes/18.783/2015/LectureNotes8.pdf)). A small
`F5` fixture independently enumerates both curves' affine points and includes
the point at infinity.

The field-order, twist-search work, and complete serialized relation size are
admitted before nonsquare search. The operation currently requires `q <=
4096`; characteristics 2 and 3, singular curves, and larger fields are
outside its contract.

[Quadratic twist model](elliptic-curve-finite-field-twists.md) ·
[Frobenius data](elliptic-curve-frobenius.md) ·
[Extension counts](elliptic-curve-extension-counts.md) ·
[Number theory operations](index.md)
