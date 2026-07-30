# Polynomial normalization

Combine like terms in the exact sparse polynomial from `input.json`. Return
canonical rational coefficients and exponent vectors, omitting zero terms.
Record the cancellation and resulting terms in `evidence/answer.txt`, and use
its SHA-256 digest in the evidence list. Write `submission.json` to the exact
agent-visible `submission_schema.json`.
Claim `VERIFIED` only by writing
`evidence/verification-record.json` to the exact agent-visible
`verification_record_schema.json` and binding it through the submission
descriptor; otherwise claim `COMPUTED`.
