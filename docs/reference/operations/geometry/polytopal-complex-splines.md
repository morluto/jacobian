# Rational polytopal complexes and splines

[Geometry operations](index.md) · [Tool surface](../../tools.md)

## Piecewise-polynomial smoothness profile

`piecewise_polynomial.smoothness_profile.compute` computes the exact largest
requested `C^r` order across every interior facet between maximal cells. For
pieces `p` and `q` meeting on the rational hyperplane `ell = 0`, the operation
checks whether `p - q` is divisible by `ell^(r+1)` over `QQ`; this is the exact
polynomial criterion for agreement of normal derivatives through order `r`.
The facet rows report `-1` when even continuity fails. The global order is the
minimum over interior facets, or the requested maximum when there are no such
facets. This reports continuity across codimension-one interfaces; it does not
claim geometric (`G^r`) continuity under reparameterization.

The divisibility criterion is standard for multivariate polynomial splines on
face-to-face partitions; see the [AlgebraicSplines package documentation](https://www.macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/AlgebraicSplines/html/index.html)
and the discussion of the classical analytic approach in this
[polytopal-splines paper](https://msp.org/pjm/2016/281-2/pjm-v281-n2-p.pdf).

`polyhedral_complex.spline_dimension.compute` returns the exact rank, nullity,
coefficient axis, and complete compatibility matrix for the same finite spline
space, without constructing its nullspace basis. This separate result is useful
when the full basis would dominate storage. It admits coefficient width at most
4096, at most 1,048,576 predicted matrix cells, at most 32,000,000 dense rank
work units, at most 32,768 decimal digits per predicted exact rank intermediate,
at most 512 MiB of intermediate matrix storage, and at most Jacobian's 10 MiB
canonical JSON output limit. A conservative canonical-JSON size estimate
includes the source complex, coefficient axis, and exact matrix scalars before
rank elimination. The row bound per facet is the dimension of the degree-at-most-d
remainder space modulo its defining linear form to power `r+1`; total dense rank
work is bounded from that row bound and the coefficient-axis width. Exact
matrix scalar heights and output size are measured before rank elimination.

The result retains the source complex and ordered `(cell ID, monomial)` axis,
so its matrix and nullity survive JSON transport with their mathematical
meaning intact. For an unconstrained one-cell space, the matrix has zero rows
and retains its full coefficient width.

## Affine transport

`polytopal_complex.affine_transform.compute` applies `x -> A*x+b` to every
cell and face of a canonical rational complex. `A` is an invertible square
rational matrix and `b` is a rational translation in the complex's existing
labelled coordinate space. The result contains the canonical transformed
complex plus bijective source-bound maps for maximal cells and faces. The maps
preserve dimensions, maximal-cell support of each face, cover relations,
pairwise intersections, and cell-facet incidence. The empty face maps to the
empty face. No source or target incidence IDs are treated as geometric
coordinates.

The operation admits ambient dimension at most four, matrix and source
coordinate components at most 32 decimal digits, at most 2,000,000
matrix-coordinate products, a conservative 512-digit transformed-coordinate
height, and at most 10 MiB of estimated source, target, and transport output
before rebuilding the geometry. The transformed coordinates must also fit the
polytopal-complex construction envelope. Singular matrices are a mathematical
domain error.

The `polytopal_complex.spline.evaluate.compute` operation evaluates one exact
rational linear combination of the canonical basis returned by
`polytopal_complex.spline.space.compute`. Supply the same closed complex,
degree, and smoothness, then give one rational coefficient for each basis row
and a point on the complex's labelled coordinate axes. The result reports all
containing cell IDs and the exact value. A point outside the complex returns
the containing-cell list empty and no value.

The operation admits degree at most 12, smoothness at most 4, and at most 4096
basis coordinates. It constructs the canonical spline basis through the same
bounded path as the spline-space operation, so the coefficients have a stable
mathematical interpretation across serialization. Coefficients and point
coordinates are exact rationals.

## Addition of compatible functions

`piecewise_polynomial.add.compute` adds two `COMPATIBLE` values on the exact
same canonical complex. It re-establishes each source's piece assignment and
shared-face continuity from the cell polynomials, then adds coefficients with
matching monomial exponents. Since this carrier represents C0 piecewise
polynomials, the result is also C0; its total degree is at most the maximum of
the two input degrees. The result retains the exact polynomial on every
maximal cell and the complete shared-face compatibility ledger. Before
arithmetic, the operation bounds the union of terms per cell, rational scalar
growth, and the compatibility-reduction work. It rejects results requiring
more than 4096 terms in any piece.

## Rational scalar multiplication

`piecewise_polynomial.scalar_multiply.compute` multiplies every cell
polynomial by one exact rational scalar. It rechecks continuity from the
polynomial pieces and returns the complete shared-face compatibility profile.
The zero scalar returns the zero function on the same complex, including its
full zero compatibility ledger. Together with addition, this supplies the
`QQ`-vector-space operations on a fixed complex and polynomial degree bound;
the linearity follows because each face-restriction condition is linear in the
piece coefficients. Admission bounds every product coefficient and the full
result before constructing scaled polynomials.

## Multiplication of compatible functions

`piecewise_polynomial.multiply.compute` takes two `COMPATIBLE` values on the
identical canonical complex and ordered rational polynomial ring. It
revalidates both source continuity ledgers, multiplies the exact polynomials
cell by cell, and returns a zero shared-face difference ledger. Restriction to
an affine face preserves products, so products of C0 functions remain C0. The
result's total degree is at most the sum of the input degrees; the operation
does not claim higher smoothness because this value type records only C0
compatibility. Before polynomial expansion, it bounds aggregate convolution
work, support, coefficient growth, compatibility work, and the complete
serialized result against the canonical 10 MiB output limit.

## Common refinement

`polytopal_complex.common_refinement.compute` takes two canonical complex
values in the same labelled rational affine space. Every maximal cell must
have the full ambient dimension; lower-dimensional support components are
rejected because ambient volume cannot certify their coverage. It intersects every pair
of maximal cells and returns the face-closed overlay together with one row
binding each overlay cell to its source-cell pair. The operation re-canonicalizes
the source maximal cells, so serialized incidence claims are not trusted as
inputs. Exact volume conservation on every source cell proves that the two
supports agree; a support mismatch is a domain error. The current admitted
slice has ambient dimension at most three, at most eight cells per input,
sixteen vertices per cell, and at most sixteen overlay cells. All coordinates,
intersections, and volumes are rational.
