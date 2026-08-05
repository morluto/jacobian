# Audit an extremal subset-sum formalization

Compare the frozen informal requirement with the proposed Lean declaration.
Determine whether the declaration faithfully preserves the fixed outer
parameter and the requirement that no subset of a candidate sums to the target.

Supply two exact certificates:

1. two cutoff multipliers for the same target whose legacy extrema disagree,
   showing that the shadowed universal binder makes one function value satisfy
   incompatible equations;
2. the exact legacy and intended extrema on the frozen finite universe,
   including a legacy-optimal candidate, an intended-optimal candidate, and a
   subset that invalidates the legacy candidate under the intended predicate.

The legacy predicate checks only the sum of the whole candidate. The intended
predicate checks every subset, including the empty subset and the candidate
itself. Use lists as mathematical sets: entries must be strictly increasing.

Do not claim that Lean parsing, elaboration, compilation, or the corrected
asymptotic conjecture has been verified. Write `submission.json` to the exact
agent-visible `submission_schema.json`. Put a concise audit in
`evidence/answer.txt`, include a `RESULT_JSON:` line containing the submitted
result as JSON, and bind that file with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, semantic scope, completeness, digest-bound evidence, limitation claims, and assurance as separate protocol dimensions. State that Lean compilation is not assessed and do not claim that Lean or the corrected asymptotic conjecture was verified or proved.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PROPOSED_FORMALIZATION_NOT_FAITHFUL`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
