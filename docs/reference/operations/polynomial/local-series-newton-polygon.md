# Newton polygons for local-series polynomials

`local_series.polynomial.newton_polygon.compute` computes the exact lower convex
hull of the points `(j, v(a_j))` for a sparse polynomial
`F(y) = sum_j a_j(t) y^j`. Each coefficient is a finite Laurent prefix over
`QQ` in one shared local parameter, place, and center. Its valuation is exact
when the retained prefix contains a nonzero coefficient. A null coefficient
row means exact zero (valuation `+infinity`); an all-zero finite prefix is
rejected because its omitted tail is unknown.

The result retains the source, every coefficient valuation, all finite points,
the hull vertices, and each edge's exact slope, horizontal length, and source
degrees lying on that edge. The hull uses integer cross products and does not
infer roots or assert a Newton--Puiseux expansion.

Admission is bounded to 256 coefficient rows, 8,192 aggregate Laurent slots,
32,768 in `y` degree, and 256 digits per rational scalar. Empty and singleton
point sets return their corresponding empty or singleton hull with no edges.

The lower Newton polygon is the lower convex hull of the valuation points; see
[these algebraic number theory notes](https://mathweb.ucsd.edu/~ebelmont/785-notes.pdf).

[Polynomial operations](index.md) · [Tool surface](../../tools.md)
