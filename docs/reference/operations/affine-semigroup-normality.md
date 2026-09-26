# Two-dimensional affine-semigroup normality

`affine_semigroup.normality.compute` decides whether a positive, full-rank
affine semigroup `S` in `Z^2` is normal in its generated group lattice. Normality
means

```text
S = cone(S) ∩ gp(S).
```

The operation returns the source semigroup and `normal=true` when equality
holds. Otherwise it returns `normal=false` and one exact hole `h` in
`cone(S) ∩ gp(S) \ S`. The witness lies in the source's ambient row axes. It is
checked for cone membership, generated-lattice membership, and semigroup
nonmembership by a complete bounded factorization search.

The kernel computes the Hilbert basis of `cone(S) ∩ gp(S)` through the
two-dimensional normalization operation, then tests each basis element in
`S`. If every Hilbert generator belongs to `S`, the whole normalization is
contained in `S`; the reverse inclusion always holds. If one does not belong,
that generator is a hole. This uses the standard identity that the integral
closure of an affine monoid in its generated group is `cone(S) ∩ gp(S)`;
Gordan's lemma makes this intersection affine. See the [Normaliz affine-monoid
reference](https://www.normaliz.uni-osnabrueck.de/wp-content/uploads/2017/01/Normaliz3.2.0Documentation.pdf),
Appendix A.4, Definition 6 and Theorem 7. Jacobian uses its own exact bounded
kernel and does not invoke Normaliz.

## Admission

The current envelope requires a positive semigroup with two ambient rows and
full rank in its generated group. The normalization cone Hilbert determinant
is at most 1,000, so its basis has at most 1,001 candidate generators. An
integral covector from the two extreme rays bounds each generator coefficient;
it avoids using the supplied rational grading for the membership search. Each
candidate coefficient box is limited to 50,000 tuples. Before searching any
box, the operation admits a total work estimate that weights each tuple by
the recursive coefficient traversal and exact `A u = h` leaf check, plus a
bound for cone/lattice witness checks. The aggregate must be at most 2,000,000
units. An exceeded bound is an operational admission error, never a normality
or hole conclusion. No degree cutoff is used to establish normality.
