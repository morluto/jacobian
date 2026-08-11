# Separate convergence in probability from almost-sure convergence

On `[0,1)` equipped with Lebesgue measure, consider the dyadic typewriter sequence. At level `k`, enumerate the indicators of all half-open intervals `[j/2^k,(j+1)/2^k)` in order, using sequence index `2^k+j`.

Certify that these indicators converge to zero in probability but not almost surely. Submit exact block summaries for every frozen level and at least three freely chosen canonical rational probe points in `[0,1)`. For each probe, give the unique hit index at every level. Explain why shrinking event mass proves convergence in probability while one hit and many misses per level prevent pointwise convergence.

The verifier recomputes all powers, masses, bounds, and hit indices. A conclusion label, one finite trace, or an appeal to the source issue is insufficient. In `result`, state the checked probability argument as `probability_argument: {"event_mass_formula":"1/2^k","limit":"ZERO"}` and the pointwise separation argument as `pointwise_argument: {"hit_count_per_level":1,"miss_count_per_level":"AT_LEAST_ONE"}`. The evidence must include exactly one `RESULT_JSON:` line whose JSON equals the submitted `result` object. The result also carries the closed `research_scope` facts `underlying_problem: NOT_ADJUDICATED` and `lean_theorem: NOT_ELABORATED`; set `limitations` to the exact label `NO_OPEN_PROBLEM_OR_LEAN_CLAIM`. These typed facts, not inferred prose, bind the mathematical argument and task scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `CONVERGENCE_MODES_SEPARATED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
