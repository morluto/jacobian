# Construct a finite invariant-subspace certificate

The frozen input defines the additive-group action

`alpha_t(x,y) = (x + t*y, y)`

on `QQ[x,y]` and a homogeneous degree-four polynomial `f`. Construct an
ordered five-element rational polynomial basis for a finite-dimensional
subspace containing `f`, together with coordinates of `f` and the exact
polynomial action matrix in that basis.

Columns encode images: if `B=(b_0,...,b_4)`, entry `(i,j)` is the coefficient
of `b_i` in `alpha_t(b_j)`. All sparse term lists must be canonical: nonzero
reduced rational coefficients, unique exponent tuples, and ascending exponent
order. The basis polynomials must be homogeneous of total degree four and
linearly independent.

Your certificate must make it possible to check all of the following without
trusting a preferred basis:

1. the submitted coordinates reconstruct the frozen `f`;
2. every substituted basis polynomial equals the corresponding matrix column;
3. `R(0)=I`;
4. `R(s+t)=R(s)R(t)` as an exact polynomial identity.

Write a concise explanation to `/app/evidence/answer.txt` distinguishing the
finite coefficient expansion from the coaction/group-law argument that proves
invariance. Do not claim the unrestricted theorem or a formal proof-assistant
certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit one non-singular degree-four basis, exact coordinates, and its full additive-group action matrix. Sparse terms use canonical reduced rationals and unique ascending exponents. In the solver's own words, the limitations array must disclose that the frozen degree-four certificate does not prove the general local-finiteness theorem.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FINITE_INVARIANT_SUBSPACE_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
