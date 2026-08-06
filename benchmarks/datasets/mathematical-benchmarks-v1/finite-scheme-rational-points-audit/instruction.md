# Audit rational points as a scheme invariant

The frozen source uses an empty scheme to show that `k`-rational points do not determine a `k`-scheme. Produce a stronger nonempty certificate over `k=F_5`.

Let `B=F_5^3` with orthogonal idempotent basis `(f0,f1,f2)`. Let
`A=F_5[u]/(u^3) × F_5 × F_5` with basis `(e0,e1,e2,u,u2)`, where the `ei` are orthogonal idempotents, `e0*u=u`, `e0*u2=u2`, `u*u=u2`, and all other products involving `u` or `u2` vanish. Units are `e0+e1+e2` and `f0+f1+f2`.

Submit both full multiplication tensors, both unit vectors, the coordinate matrix of the unital map `B -> A` sending `fi` to `ei`, and **all** unital algebra maps from each algebra to `F_5` as image vectors on the ordered bases. Give the induced map on rational points by precomposition. Finally exhibit a nonzero nilpotent of exact order three in `A`, while certifying that `B` has no nonzero nilpotent.

The verifier independently enumerates every possible linear functional (`5^5` for `A`, `5^3` for `B`) and checks multiplicativity on all basis pairs. It also rebuilds every product and power. A bijection on the three nonempty rational-point sets therefore coexists with a reducedness obstruction to algebra—and hence affine-scheme—isomorphism.

The evidence file `evidence/answer.txt` must state the following three facts in the solver's own words: (1) both affine schemes are nonempty and have the same three rational points under the induced map; (2) A has a nonzero order-three nilpotent; (3) B is reduced, so the two schemes are not isomorphic. Additional derivation content is allowed and ignored.

The public submission contract is generated below from `tests/public_contract.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit complete finite-algebra structure constants, exhaustive rational-point lists, the induced point bijection, and a reducedness-separating nilpotent certificate. The verifier reconstructs every operation over F_5. In the solver's own words, the limitations array must disclose that this is one finite affine countermodel over F_5, not a general theorem about schemes or functors of points.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `RATIONAL_POINT_BIJECTION_DOES_NOT_FORCE_SCHEME_ISOMORPHISM`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
