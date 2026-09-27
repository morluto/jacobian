# Solvable radical of a finite-dimensional Lie algebra

`lie_algebra.solvable_radical.compute` returns the largest solvable ideal
`rad(g)` of a finite-dimensional Lie algebra over `QQ`, as the existing
source-bound `LieIdeal` value. The basis rows are canonical RREF coordinates on
the input algebra's ordered basis, so the value composes directly with ideal
checks and quotient construction.

In characteristic zero the solvable radical is the Killing-orthogonal
complement of the derived algebra:

\[
\operatorname{rad}(\mathfrak g)=[\mathfrak g,\mathfrak g]^\perp,
\qquad
\mathfrak g'=[\mathfrak g,\mathfrak g].
\]

The operation computes the exact Killing matrix
`K[i,j] = tr(ad(b_i) ad(b_j))`, constructs the exact derived subspace, and
returns the right nullspace of its pairing with `K`. It validates the Lie
bracket once, bounds coefficient growth, exact linear-algebra work, canonical
result height, and serialized bytes before each matrix expansion. This
characteristic-zero identity is stated and proved in Nathan Jacobson,
[*Lie Algebras*](https://openlibrary.org/books/OL5829050M/Lie_algebras.),
Chapter III, Theorem 5 (Interscience Publishers, 1962).

The distinction from the radical of the Killing form matters: for the affine
algebra `[h,e]=e`, the solvable radical is all of `g`, while the Killing-form
radical is only the line spanned by `e`.

The admitted algebra carrier is finite-dimensional over `QQ`, with dimension
at most eight and bounded rational structure constants. Requests whose
intermediate exact matrices or canonical result exceed this operation's
coefficient, work, or four-megabyte output envelope are rejected before the
corresponding exact expansion.

[Operation references](index.md) · [Tool surface](tools.md)
