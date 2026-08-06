# Subset-incidence determinant factorization

This Regression-family task freezes Putnam 2018 A2 from PutnamBench's
Apache-2.0 Lean source at commit
`dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c`.

Its single objective is combinatorial matrix factorization: discover the
inclusion-exclusion diagonalization and convert it into a determinant parity
formula.  Difficulty is **Hard (provisional)** because it requires changing to
the subset-zeta basis, not expanding a 31-by-31 determinant.

Shortcut audit: the two public determinant values, direct determinant calls,
and reordered sign lists fail without the exact incidence transform and full
parity trace.  The verifier reconstructs all matrices with integer arithmetic
and reports `COMPUTED`; it does not replay the universal Lean theorem.
