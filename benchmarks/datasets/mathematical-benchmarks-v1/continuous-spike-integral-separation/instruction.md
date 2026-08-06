Construct a strictly positive continuous function on `[1,+infinity)` for which
the improper integral diverges but the series of integer samples converges.

Submit the exact first twelve spike supports and areas, the twelve integer
samples, and the general symbolic series classifications. Each spike support
must be disjoint from every other support and must avoid every integer. The
spike areas must form a divergent series while the integer samples form a
convergent series.

Your certificate at `/app/evidence/answer.txt` must show that the supports are
disjoint and avoid every integer, the spike areas form a divergent series, and
the samples form a convergent series. Write `/app/submission.json` and
digest-bound `/app/evidence/answer.txt`. Each rational field in the submission
must be a canonical reduced fraction string of at most 100 characters, and the
evidence file must be at most 16 MiB. Claim at most `COMPUTED`; `UNVERIFIED` is
also accepted.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `VALID_INTEGRAL_SERIES_SEPARATOR`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE_FOR_DECLARED_FAMILY`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
