# SAT decision with witness

Decide the exact CNF formula in `input.json`. For a satisfiable formula, return
`SATISFIABLE` and a complete Boolean assignment that satisfies every clause.
For an unsatisfiable formula, return `UNSATISFIABLE` only with the evidence
appropriate to that conclusion. Record the clause-by-clause check in
`evidence/answer.txt`, include its SHA-256 digest, and write `submission.json`
to the exact agent-visible `submission_schema.json`. Claim `VERIFIED` only by
writing `evidence/verification-record.json` to the exact agent-visible
`verification_record_schema.json` and binding it through the submission
descriptor; otherwise claim `COMPUTED`.
