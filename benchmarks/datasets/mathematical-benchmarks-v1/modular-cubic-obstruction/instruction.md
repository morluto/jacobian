# Certify a modular obstruction

Prove the universal integer claim in the offline input using modulus 7 and a
complete residue certificate that rules out the equation.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
The residue table must cover every possible residue of `x` modulo the submitted
modulus exactly once. Put a concise derivation in `evidence/answer.txt`, include
a `RESULT_JSON:` line containing the submitted result as JSON, and bind that file
with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
