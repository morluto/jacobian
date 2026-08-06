# Separate convergence in probability from almost-sure convergence

On `[0,1)` equipped with Lebesgue measure, consider the dyadic typewriter sequence. At level `k`, enumerate the indicators of all half-open intervals `[j/2^k,(j+1)/2^k)` in order, using sequence index `2^k+j`.

Certify that these indicators converge to zero in probability but not almost surely. Submit exact block summaries for every frozen level and at least three freely chosen canonical rational probe points in `[0,1)`. For each probe, give the unique hit index at every level. Explain why shrinking event mass proves convergence in probability while one hit and many misses per level prevent pointwise convergence.

The verifier recomputes all powers, masses, bounds, and hit indices. A conclusion label, one finite trace, or an appeal to the source issue is insufficient. The explanation must agree with the submitted result: include exactly one `RESULT_JSON:` line whose JSON equals the submitted `result` object, and articulate the universal pointwise argument (every point lies in one interval per level, so the sequence equals one and zero infinitely often at that point). Do not claim that the underlying open problem or Lean theorem is solved or machine verified; a limitation must state this restriction in unambiguous negated language rather than merely mentioning the open problem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `CONVERGENCE_MODES_SEPARATED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
