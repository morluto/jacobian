# Repair a parameterized sharp-bound proof

For positive `a,b,c,d` with `a+b+c=1`, audit the frozen claim about the sharp
lower bound of `a^3+b^3+c^3+abc*d`.

1. Determine the exact transition value of `d` and both sharp-bound formulas.
2. Distinguish the regime where equality is attained at the symmetric point
   from the regime where the sharp value is only a boundary infimum.
3. Supply the convex-interpolation coefficients joining the `d=0` tangent
   certificate to the transition Schur certificate.
4. Supply the nonnegative remainder coefficient above the transition.
5. Give the symmetric equality point and a valid permutation of the limiting
   family with one variable tending to zero.
6. Identify the concrete defect in the frozen explanation.

Write `submission.json` to the exact schema. Put a concise repair in
`evidence/answer.txt`, include one `RESULT_JSON:` line containing the exact
submitted result as compact JSON, plus one `BOUNDARY_FAMILY_JSON:` line
containing the exact submitted `boundary_family` object as compact JSON, and
bind that file with its SHA-256 digest.
Use `COMPUTED`, not `VERIFIED`: the checker replays this bounded symbolic
certificate but is not an external proof assistant.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `PIECEWISE_BOUND_REPAIRED`, `FROZEN_EXPLANATION_VALID`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
