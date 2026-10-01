# Rational polytopal complexes and splines

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

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
work units, at most 32,768 decimal digits per predicted exact intermediate,
at most 512 MiB of estimated intermediate matrix storage, and at most
10,485,760 decimal digits summed over the stored rational matrix components.
These are native computation and representation bounds, not JSON byte limits.
All are admitted from the source before polynomial-power construction,
remainder construction, or rank elimination. The row bound per facet is the dimension of
the degree-at-most-d remainder space modulo its defining linear form to power
`r+1`; dense rank work uses that bound and the coefficient-axis width.

For coefficient growth, clear the facet form's denominators and primitive
content to write it as `A*x + B(y)`. Monomials with x-exponent below `k=r+1`
are unchanged. For exponent `m >= k`, the x^j coefficient of the remainder has
magnitude
`binom(m,j) * binom(m-j-1,k-j-1) * (B/A)^(m-j)`, for `0 <= j < k`.
The coefficient one-norm of `B` bounds its powers, and `A^m` clears every
remainder denominator. A global column combines denominator bounds from all
incident facets; its unit entries are scaled too. Unit columns remain cheap
only when the entire column has denominator one.

The kernel constructs this closed form directly in the original ambient axes.
It caches integer powers of `B` and `A`, then each x-power remainder once per
facet. Linear convolution has partial coefficient magnitude at most
`||B||_1^t`; a Taylor piece multiplies such a coefficient by the displayed
binomial factor before constructing its rational coefficient. Different
x-exponents have disjoint support, so there are no rational coefficient sums,
polynomial divisions, or expanded divisors. If `k>d`, all remainders are
identity monomials and no powers are built.

Rank uses FLINT fraction-free elimination after clearing denominators by
column; the returned compatibility matrix keeps its original rational entries.
A minor bound sums the largest column heights and the factorial term instead
of charging every column the widest coefficient. This admits dense rational
four-dimensional facets with 32-digit source coordinates at degree 4,
smoothness 3, and the larger degree-5, smoothness-4 control. Power/piece
construction, stored coefficients, and elimination temporaries are bounded
separately. The latter includes the two-product subtraction before Bareiss
exact division, not only the final minor height. Profiles outside these
derived envelopes receive a typed resource refusal before expansion.

The 512 MiB limit is an explicit accounting budget, not a process-RSS guarantee.
The model charges one byte per decimal digit, 64 bytes per integer,
64 per Fraction wrapper, 512 per canonical-scalar wrapper, 8 per container
reference, and 64 per row/container header. It sums the rational and canonical
matrices, Python scaled-integer construction list, FLINT input and separate LU
matrices, row containers, coefficient axis, column state, permutations, source
bounds, and active facet caches. Cache terms include mapping/exponent overhead;
the convolution accumulator and support sorting are also charged. Thirty-two
maximum-height integer scratch slots cover bounded scalar arithmetic. All
matrix representations are charged together even where their lifetimes do not
overlap. Source-complex storage stays under its existing owner admission;
Allocator/runtime overhead outside these charged payloads is not a promised
process-RSS ceiling. Newly added loops observe request checkpoints.

The closed form also repairs an axis-order defect: division with an inactive
first ambient variable could leave terms such as `x*y` unreduced modulo
`y-1`. The existing row-bound check rejected these spaces; two vertically
stacked squares now give the same spline dimensions as the
axis-permuted fixture. This is an acceptance repair: standard remainder columns
already occupy the exact per-facet row bound, so any additional non-normal
support previously forced that check to fail. Previously accepted normal
remainders retain their unique coefficients and ambient axis ordering.

The result retains the source complex and ordered `(cell ID, monomial)` axis,
so its matrix and nullity survive JSON transport with their mathematical
meaning intact. For an unconstrained one-cell space, the matrix has zero rows
and retains its full coefficient width.

The native `piecewise_polynomial_global_profile` helper checks whether a
continuous piecewise polynomial on full-dimensional maximal cells is the
restriction of one ambient polynomial. It is deliberately not a catalog
operation: selecting the first cell polynomial and comparing the rest is a
cheap deterministic projection of the existing piecewise-polynomial value.
Restriction to a full-dimensional cell is injective, so exact coefficient
comparison proves or refutes the profile. Lower-dimensional maximal cells do
not determine a unique ambient polynomial and are rejected.

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

The `polyhedral_complex.spline.evaluate.compute` operation evaluates one exact
rational linear combination of the canonical basis returned by
`polyhedral_complex.spline_space.compute`. Supply the same closed complex,
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
