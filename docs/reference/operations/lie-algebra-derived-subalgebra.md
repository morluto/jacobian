# Lie algebra derived subalgebra

`lie_algebra.derived_subalgebra.compute` returns

\[
  [\mathfrak g,\mathfrak g]
  = \operatorname{span}_{\mathbb Q}\{[b_i,b_j]\}
\]

as a canonical RREF `LieIdeal` in the source algebra's ordered basis. The
value retains the ambient algebra, including when the derived ideal is zero.
It composes with ideal consumers such as exact quotient construction.

The operation first admits the finite-dimensional rational structure
constants and establishes antisymmetry and every basis-triple Jacobi identity.
It then brackets basis vectors and reduces the resulting exact coordinate
vectors. Jacobi proves that `[g,g]` is an ideal, so the `LieIdeal` result is
sound without replaying bracket calculations in result construction.

For the three-dimensional Heisenberg algebra with `[x,y]=z`, the returned
ideal has the single RREF row `(0,0,1)`. For an abelian algebra it is the
zero ideal with the original ambient basis retained.

Sage's finite-dimensional Lie algebra APIs provide exact derived algebras and
derived series from structure coefficients ([Sage reference](https://doc.sagemath.org/html/en/reference/categories/sage/categories/finite_dimensional_lie_algebras_with_basis.html)).
