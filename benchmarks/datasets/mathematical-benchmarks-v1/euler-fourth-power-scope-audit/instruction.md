# Euler fourth-power claim scope audit

Refute the frozen universal claim by supplying a primitive positive-integer witness
`x^4 + y^4 + z^4 = w^4` within the declared bound. The three left bases must be
strictly increasing, all four bases must be distinct, and their joint gcd must be one.

Submit the exact fourth powers, both sides of the equality, and residue checks for
every required modulus. Explain why this single witness refutes exactly the frozen
claim, while establishing neither minimality nor any statement about other exponents
or numbers of summands.

The verifier accepts any witness satisfying this contract; it does not compare against
one memorized tuple. Claim at most `COMPUTED`. Write `/app/submission.json` and bind
`/app/evidence/answer.txt` by SHA-256.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FALSE`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
