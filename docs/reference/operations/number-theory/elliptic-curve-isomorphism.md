# Finite-field elliptic model isomorphism

`elliptic_curve.finite_field.isomorphism.decide` compares two nonsingular
short-Weierstrass models over the same exact finite-field presentation. In the
admitted characteristic greater than three, every isomorphism between these
models has the form

\[
(x,y) \longmapsto (u^2x,u^3y), \qquad u\ne 0.
\]

The operation searches the complete nonzero field and returns either an
explicit scaling element or `null` when no scaling exists. A negative result is
therefore an exact decision within the admitted field bound, not a heuristic.
The input must use the same canonical field presentation on both curves. The
complete search admits field order at most 4096 and enforces a work bound before
searching.

This compares model isomorphism only. Isogeny is a separate relation and is
reported by [`elliptic-curve-isogeny-class`](elliptic-curve-isogeny-class.md).
