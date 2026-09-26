# Exact tropical operations

The tropical domain supports distinct `MIN_PLUS` and `MAX_PLUS` semirings over
exact `ZZ` and `QQ` values. Scalar, vector, sparse polynomial, matrix,
assignment, and finite matrix-power operations return exact results. These
contracts concern formal tropical polynomials and finite computations; they do
not imply functional equality or solve tropical systems.

## Scalar duality

`tropical.scalar.dual.compute` maps a scalar between min-plus and max-plus by
ordinary negation. It maps `+infinity` to `-infinity` and conversely, preserves
the exact `ZZ` or `QQ` base, and returns both source and target semiring
identities with the mapped scalar. Applying the map twice recovers the original
value. It accepts only the infinity licensed by the source semiring; the
two-infinity completion and its indeterminate addition cases are not part of
this contract. The isomorphism by negation is the standard min-plus/max-plus
duality described in [From max-plus algebra to nonexpansive mappings](https://www.jeremy-gunawardena.com/papers/fmpatnm.pdf).

## Vector projectivization

`tropical.vector.projectivize.compute` takes a vector with a unique labelled
axis. For min-plus it subtracts the minimum finite coordinate; for max-plus it
subtracts the maximum finite coordinate. The result retains the axis and
infinity support, has normalized finite extremum zero, and returns the exact
common translation that was added. The all-infinity vector returns
`NO_PROJECTIVE_CLASS`, since it has no finite coordinate by which to normalize.

The operation is exact for integer or rational coordinates. Its work is linear
in the vector dimension (at most 128); input and translated scalar sizes are
checked against the shared 8,192-digit scalar envelope before result
construction. Adding the returned translation to every finite output
coordinate minus the returned translation reconstructs the source vector. The normalized representative is
unique in its translation class under the selected extremum convention.

## Formal polynomial powers

`tropical.polynomial.power.compute` raises one sparse formal polynomial to a
nonnegative integer power at most 16. Exponent zero returns the multiplicative
identity, the constant polynomial with coefficient zero and zero exponent on
the same variable axis. Positive powers use exact sparse convolution, with
intermediate support, exponent, scalar-growth, total-work, and serialized
output admission completed before coefficient multiplication. The estimated
serialized result must fit Jacobian's 10 MiB canonical output limit.

The result is the normalized formal exponent/coefficient map. It does not
remove monomials that are inessential as piecewise-linear functions and does
not establish functional equality. For example, the cube of `0 ⊕ x` in
min-plus has formal terms `0 ⊙ x^k` for `k = 0, 1, 2, 3`.

## Simultaneous polynomial substitution

`tropical.polynomial.substitute.compute` maps each variable of a sparse source
polynomial to a sparse tropical polynomial over one explicit target variable
axis. Images are positional: image `i` substitutes the source variable at
`polynomial.variables[i]`. This is simultaneous substitution, so variables in
an image are interpreted on the target axis and are not substituted again.
The returned value is the exact normalized formal polynomial, not a functional
normal form.

The operation preflights exact Minkowski support for every source monomial,
including all binary-power intermediates, before coefficient arithmetic. Each
intermediate and final polynomial has at most 512 terms; coordinate exponents
are at most 1,024; total sparse pair work is at most 750,000 pairs and 10
million coordinate additions; coefficient growth is at most 8,192 decimal
digits; and the estimated canonical result fits the 10 MiB output limit.
Substitution into a zero image removes any source monomial that uses that
variable with positive exponent. These bounds apply to the expanded formal
result, so a small functional simplification does not license an oversized
intermediate.

For example, substituting `0 ⊕ t` for `x` in `0 ⊕ x` over min-plus returns
`0 ⊕ t`; the repeated constant exponent is merged by taking its tropical
minimum.

## Tropical assignment profiles

`tropical.matrix.assignment_profile.compute` accepts an `n × n` tropical
matrix, with `n ≤ 8`. Row and column axes are distinct labelled sets; they
must have equal lengths, but their labels need not agree. Each returned
permutation is an `n`-tuple of column indices, one per row in source row-axis
order. The result gives the minimum assignment weight in min-plus or maximum
assignment weight in max-plus, together with every tied optimal permutation.
It is an assignment optimum, not an ordinary signed determinant. The fixed
eight-row bound admits at most `8!` candidate permutations before enumeration.
If every permutation uses at least one infinite entry, the optimum is the
semiring additive identity and every permutation is returned as tied.

## Univariate polynomial roots

`tropical.polynomial.univariate_roots.compute` accepts exactly one variable.
Each finite coefficient is the intercept of an affine function, and its
exponent is that function's slope. The min-plus polynomial is their lower
envelope; the max-plus polynomial is their upper envelope. The operation
returns the complete piecewise-linear profile on open intervals and every
finite breakpoint as an exact rational. At a breakpoint it reports all tied
exponents, including a term that is active only at that point, the adjacent
active exponents and slopes, and multiplicity equal to the absolute slope
jump. Thus min-plus slopes weakly decrease and max-plus slopes weakly increase
across roots.

A nonzero constant or monomial has one interval and no finite roots. The zero
polynomial is represented explicitly as `ZERO_POLYNOMIAL` with an empty
profile; no infinite affine root is introduced. Crossover pairs and a
conservative serialized-result byte estimate are admitted before exact
intersection arithmetic. Root numerators and denominators have a dedicated
16,384-digit limit, and the output is capped at 16 MiB.

## Univariate split form

`tropical.polynomial.univariate_split_form.compute` returns the consecutive
support polynomial obtained by multiplying the linear factors determined by
the finite roots and their slope-jump multiplicities. It preserves the
induced piecewise-linear function, including when the source has inessential
terms, but does not claim formal polynomial equality. An integer-coefficient
input is promoted to `QQ` when its split coefficients require rational values.
The operation admits consecutive support before root computation and checks
exact scalar growth before each coefficient sum. The output retains at most
512 terms, with each coefficient bounded by the shared 8,192-digit scalar
envelope. This is the univariate split form described in the [combinatorial
introduction to tropical geometry](https://math.berkeley.edu/~bernd/tropical/sec1.pdf).
