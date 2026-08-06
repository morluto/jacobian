Construct an optimal set of answer strings for the frozen six-question,
three-option problem in `/app/input.json`.

Submit `/app/submission.json` and `/app/evidence/answer.txt`. Your JSON must:

1. give 18 distinct ternary words of length six, with every distinct pair
   agreeing in exactly zero or two coordinates;
2. report the exact average distance distribution `A0`, `A4`, and `A6`;
3. compute the order-1 and order-2 q-ary Krawtchouk values at distances
   0, 4, and 6 from the definition
   `K_j(i)=sum_h (-1)^h (q-1)^(j-h) C(i,h) C(n-i,j-h)`;
4. give nonnegative rational multipliers for those two Delsarte inequalities
   whose linear combination is exactly `18 - (1 + A4 + A6) >= 0`.

The verifier accepts any valid construction and recomputes all finite
arithmetic. A published value of 18 without the construction and dual
certificate is insufficient. Do not claim `VERIFIED`; no independent formal
proof checker is used.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `OPTIMUM_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
