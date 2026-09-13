# Gaussian-rational projective-line cross-ratios

`geometry.projective_line.cross_ratio.gaussian_rational.compute` returns the
exact cross-ratio of an ordered quadruple of pairwise distinct points of
`P^1(Q(i))`. Here `Q(i)` is the fixed Gaussian-rational field with
`i^2 = -1`; each point has the ordered homogeneous axes `(X, Y)` and is
stored in first-nonzero-coordinate normal form. The point at infinity is the
canonical point `[1 : 0]`.

The operation uses the convention

```
[p1, p2; p3, p4] = det(p1, p3) det(p2, p4)
                 / (det(p1, p4) det(p2, p3)).
```

For affine representatives `[x : 1]`, this is
`(x1 - x3)(x2 - x4) / ((x1 - x4)(x2 - x3))`. A harmonic quadruple such as
`([0:1], [1:0], [1:1], [-1:1])` therefore returns `-1`.

The determinant ratio is unchanged when each representative is rescaled by a
nonzero Gaussian rational, and when one invertible `2 × 2` Gaussian-rational
matrix is applied to all four points: every determinant receives the same
matrix determinant, which cancels. Permutations produce the usual six
cross-ratio transforms (`lambda`, `1/lambda`, `1-lambda`,
`1/(1-lambda)`, `lambda/(lambda-1)`, `(lambda-1)/lambda`).

Input points are normalized before the operation. All six pairwise
determinants must be nonzero, so coincident points are rejected before the
quotient is formed. The exact kernel has an operation-owned coefficient-height
admission for determinant products and quotient intermediates; this is a
semantic bound independent of transport byte limits. The result is the
canonical domain-owned `GaussianRational` value, so its realness is tested by
the native projection `result.imaginary == 0` rather than by a duplicate
classifier field.

The motivating source discusses cross-ratios as invariants of ordered points
under `PGL_2`; see Faber, Pardue, and Zelinsky, [*Cross-Ratios of
Scheme-Valued Points*](https://arxiv.org/abs/2012.03073).
