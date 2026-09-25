# Affine-semigroup holes through degree

[`affine_semigroup.holes_through_degree.compute`](../tools.md) returns the complete
set

```text
{ h in cone(S) intersect gp(S) : degree(h) <= D and h not in S }
```

for one positive affine semigroup `S` in two ambient dimensions and an
inclusive integer degree bound `D`. The generated-lattice context and ambient
row axes remain attached to the source semigroup. A negative `D` has an empty
profile. A bounded profile does not claim to return all global holes.

The normalization of an affine monoid in its generated group is
`cone(S) intersect gp(S)`; this is the geometric characterization in
[Normaliz documentation, Appendix A.4, Definition 6 and Theorem 7](https://www.normaliz.uni-osnabrueck.de/wp-content/uploads/2017/01/Normaliz3.2.0Documentation.pdf).
The operation uses Jacobian's exact generated-lattice coordinates. In those
coordinates, the cone cut by the positive degree bound is a rational triangle.
Its axis-aligned integer box contains every candidate lattice point. The
operation admits that box before enumeration, then computes the finite
semigroup closure inside it. Each nonzero generator has strictly positive
degree, so every semigroup point in the triangle is reached from zero by
successive generator additions that remain in the triangle.

The current execution envelope is rank two and full rank, at most ten
generators, at most 64 digits for the degree and grading components, generated
lattice coordinates under the existing eight-digit Hilbert envelope, at most
50,000 points in the containing box, at most 2,000,000 estimated work units,
and at most 2,000,000 estimated result bytes. Admission uses the box point
count, number of distinct generator vectors, exact output-coordinate bounds,
and retained source label and grading sizes. Requests outside these bounds are
resource refusals, never empty profiles.

For generators `(2,0)`, `(0,2)`, `(1,1)`, `(1,0)` with degree `x+y`, the
generated lattice is `Z^2`. Every point with positive first coordinate is in
the semigroup; on the vertical axis only even points are generated. Thus the
holes through degree four are `(0,1)` and `(0,3)`.
