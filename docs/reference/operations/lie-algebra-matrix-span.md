# Constructing a Lie algebra from a matrix span

`lie_algebra.from_matrix_span.compute` accepts an ordered, linearly independent
family of square matrices over `QQ`. The span must already be closed under the
matrix commutator `[A, B] = AB - BA`. The operation computes the induced
structure constants and returns them with the original matrix basis in matching
order. It rejects a dependent family and a span that is not closed; it does not
adjoin commutators to generate a larger subalgebra.

The exact result records the transport explicitly: basis label `M0` is realized
by the first input matrix, `M1` by the second, and so on. The matrix commutator
inherits antisymmetry and Jacobi from the associative matrix algebra, and the
independent basis makes its coordinate representation unique.

The bounded envelope is matrix order at most 8, span dimension at most 8, and
at most 64 decimal digits in each input matrix component. The complete family
of pairwise commutators, exact elimination and coordinate recovery, rational
growth, and output size are admitted before those computations begin.

The operation follows the standard matrix Lie algebra construction: square
matrices over a field form a Lie algebra under the commutator bracket. See
[Mathlib's matrix Lie algebra reference](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Algebra/Lie/Matrix.html).
