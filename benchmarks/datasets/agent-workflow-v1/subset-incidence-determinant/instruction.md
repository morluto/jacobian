# Factor the nonempty-intersection matrix

For the matrix indexed by nonempty subsets of `[n]`, with entry one exactly
when two subsets intersect, derive its determinant for every `n>=1`.

Submit an exact incidence factorization.  Choose and describe a valid
`sample_n`, provide a cardinality-then-mask ordering of all nonzero bitmasks and
the diagonal weights in that basis.  Also provide the determinant parity trace
for every `n` through `trace_max_n`, including the number of nonempty even-cardinality
subsets and the resulting determinant.

The public determinant values alone are insufficient.  The verifier rebuilds
the intersection matrix and checks the submitted factorization exactly, then
recomputes the parity formula.  Bind `/app/evidence/answer.txt`; do not claim
`VERIFIED`.
