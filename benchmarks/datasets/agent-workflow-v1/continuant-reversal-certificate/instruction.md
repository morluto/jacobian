# Certify reversal symmetry of a variable-coefficient recurrence

Let `u_0=u_1=v_0=v_1=1`, `u_(k+1)=u_k+a_k*u_(k-1)`, and `v_(k+1)=v_k+a_(n-k)*v_(k-1)`. Produce a symbolic tiling certificate explaining why `u_n=v_n` for arbitrary commuting coefficients.

For the frozen board length `n=10`, list every square-free monomial support with no adjacent indices that occurs in each final recurrence polynomial. Then give the complete reflection pairing induced by `i -> n-i`, together with the general recurrence contract.

The verifier independently enumerates the monomials and reflection map. Numeric coefficient substitutions, an answer-only equality, or an unsupported `VERIFIED` claim are insufficient. Include a line beginning `RESULT_JSON:` followed by the exact JSON object used for `result.proof_obligations`; this is the structured proof certificate the verifier replays.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `REVERSAL_SYMMETRY_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
