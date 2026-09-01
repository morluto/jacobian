# Prime-field homogeneous fixed subspaces

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`finite_field.prime_linear_action.homogeneous_fixed_subspace.compute` computes
the simultaneous fixed subspace of one homogeneous polynomial component under
explicit invertible matrices over `GF(p)`. It is the bounded linear-algebra
primitive needed by modular invariant-ring calculations; it does not construct
an invariant ring or draw a Noether-number conclusion.

The request binds an ordered polynomial-variable axis to one or more matrices.
Matrix column `j` contains the coefficients of the image of variable `j`. Thus,
for variables `(x, y)`, the matrix `[[1, 1], [0, 1]]` means
`x -> x` and `y -> x + y`. All matrices are square over the same word-sized
prime field and must be invertible. The listed matrices need only generate the
action being studied; no group presentation or closure enumeration is required
because a polynomial is fixed by the generated group exactly when it is fixed
by every supplied generator.

For degree `d`, the result lists every exponent vector of total degree `d` in
descending lexicographic order. Each row of `basis_matrix` is a coefficient
vector on that monomial basis. The rows are in reduced row-echelon form, giving
a deterministic basis independent of private FLINT nullspace choices. The
result retains the complete source action and degree, including when the fixed
space is zero-dimensional, so it can be serialized and passed back to the same
or later degree computations without reconstructing field or axis context.

Admission is derived before polynomial expansion. The variable and generator
axes are bounded by the shared prime-field matrix axis, and the homogeneous
monomial basis is bounded by that same axis. The accepted degree is then
constrained by the induced-matrix cell bound, stacked-equation axis and cell
bounds, Python substitution work, prime-field elimination work, and the exact
canonical result-size envelope. These are execution and codomain bounds, not
restrictions on the mathematical definition.

The defining source regression uses the two displayed matrices for the
five-dimensional `D8` action on `Q` in equation (4) of Muhammad Fazeel Anwar,
*Counterexamples to Conjectures of Wehlau on Noether Numbers*,
[arXiv:2607.18585v2](https://arxiv.org/abs/2607.18585). It reproduces the fixed
dimensions `(1, 2, 4, 7, 15, 23, 37, 53)` in degrees zero through seven. The
larger invariant-generation and indecomposable-quotient calculations remain
separate potential operations.
