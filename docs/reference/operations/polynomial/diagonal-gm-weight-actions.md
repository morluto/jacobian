# Diagonal multiplicative-group actions

`algebraic_group.gm.diagonal_weight_action.compute` acts on one exact ordered
polynomial ring over `QQ` by a closed diagonal integer-weight action

\[
\lambda\cdot x_i=\lambda^{w_i}x_i.
\]

The input binds one integer weight to every variable. For a source monomial
`c*x^e`, the operation returns the Laurent coaction term
`c*x^e*t^(sum_i w_i*e_i)`. The output retains the source parent, the complete
Laurent coaction, every nonzero integer-weight component, and the weight-zero
polynomial. Negative weights are represented by the existing exact sparse
Laurent value; no coefficient field extension is needed.

The accepted envelope is at most 8 variables, weights of magnitude at most 64,
256 source terms, total degree at most 64, and 128 decimal digits per rational
coefficient. The induced Laurent exponent is bounded by 4096. These checks run
before coaction terms and projections are constructed. Result terms cannot
exceed the source term count.

For every monomial, the exponent is the dot product `w.e`. Setting `t=1`
therefore gives the counit identity. Under multiplication of parameters, the
same term becomes `s^(w.e)*t^(w.e)`, equal to `(s*t)^(w.e)` for positive,
zero, or negative integer exponents; this is the coassociativity law. The
weight-zero projection is exactly the fixed polynomial subspace for this
action. This operation handles one supplied polynomial and does not generate
an invariant ring or classify general `G_m` actions.

## Invariants through a bounded degree

`algebraic_group.gm.invariants_through_degree.compute` constructs the exact
weight-zero subspace of `QQ[x_1,...,x_n]_{<=d}`. It returns the canonical
weight-zero monomial basis and the number of invariant monomials in each exact
degree from zero through `d`. The complete candidate monomial count
`binomial(n+d,d)` is checked against 4096 before exponent tuples are generated.
Each returned basis polynomial is a normal polynomial-ring value and can be
passed directly to `algebraic_group.gm.diagonal_weight_action.compute`; its
coaction parameter has exponent zero.

This follows the standard equivalence between representations of the
multiplicative group and integer-graded modules: the weight-zero subspace is
the invariant subspace, and multiplication adds weights. See SGA 3,
Exposé I, §§4.7.3–4.7.4 for diagonalizable-group gradings and weight
projectors, and Michel Brion, *Introduction to actions of algebraic groups*,
§2.2 for the `G_m`/`Z`-grading correspondence and invariant subspace.
[SGA 3](https://grothendiecksga.com/read/sga3/en/I-4.html) ·
[Brion's notes](https://www-fourier.univ-grenoble-alpes.fr/~mbrion/ihs_final.pdf)

The grading/coaction convention follows the standard description of a
`G_m`-action on an affine scheme from a `Z`-graded coordinate ring; see
[Stacks Project, Example 39.12.3](https://stacks.math.columbia.edu/tag/0EKJ).

[Polynomial operations](index.md) · [Tool surface](../../tools.md)
