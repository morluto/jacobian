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

`local_series.polynomial.newton_edge_characteristic.compute` selects one of
those edges and returns its characteristic polynomial over `QQ`. If its left
endpoint has degree `j0`, each on-edge source term `a_j(t)y^j` contributes
`lc(a_j) c^(j-j0)`. The result retains the source, edge, and transported
leading coefficient for every term. Its nonzero roots are candidate leading
coefficients under the edge valuation substitution.

`local_series.polynomial.newton_edge_characteristic_roots.compute` returns all
roots with multiplicities when the edge polynomial has degree at most two.
Rational roots remain rational; quadratic algebraic roots use the existing
exact algebraic-root values. These are candidate leading coefficients, not
lifted branches.

`local_series.polynomial.smooth_branch_first_jet.compute` handles a separate
smooth case: the caller supplies a rational simple root `c` of `F(0,y)`, and
the operation returns `y(t)=c-F_t(0,c)/F_y(0,c)*t+O(t^2)`. It requires the
coefficient prefixes through exponent one. It does not lift ramified, multiple,
or algebraic-coefficient branches, or continue beyond the first jet.

Admission is bounded to 256 coefficient rows, 8,192 aggregate Laurent slots,
32,768 in `y` degree, and 256 digits per rational scalar. Empty and singleton
point sets return their corresponding empty or singleton hull with no edges.

The lower Newton polygon is the lower convex hull of the valuation points; see
[these algebraic number theory notes](https://mathweb.ucsd.edu/~ebelmont/785-notes.pdf).

[Polynomial operations](index.md) · [Tool surface](../../tools.md)
