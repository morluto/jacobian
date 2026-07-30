# Finite partition and exact coverage

Partition the exact universe in `input.json` into named cases according to the
given residue relation. Return every member exactly once. State `TRUE` only
when the cases are pairwise disjoint and cover the complete supplied universe.
Write `submission.json` to the exact schema in the agent-visible
`submission_schema.json`. Put the coverage calculation in `evidence/answer.txt`
and include that file's SHA-256 digest in the evidence list.
Claim `VERIFIED` only by writing
`evidence/verification-record.json` to the exact agent-visible
`verification_record_schema.json` and binding it through the submission
descriptor; otherwise claim `COMPUTED`.
