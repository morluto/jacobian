# Audit Lean semantic-guard findings across scope and pathological types

For every frozen case in `/app/input.json`, determine the exact semantic-trap
findings that a sound checker should emit.  Preserve the input order and attach
the controlling reason for each decision.

This is an assurance-calibration task, not a request to run Lean or copy the
source repository's expected messages.  Your audit must distinguish full proof
state scope from textual binder order, ordinary numeric types from a custom
type where `0` and `1` are definitionally equal, proof terms from theorem
statements, and division-by-zero risk from precision-loss truncation.

Bind `/app/evidence/answer.txt`, include a `RESULT_JSON:` line containing the
submitted result as JSON, and bind that file with its SHA-256 digest.  The
offline verifier derives every expected finding from the frozen semantic facts.
Do not claim `VERIFIED` or claim that Lean compilation was replayed.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `AUDIT_COMPLETE`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
