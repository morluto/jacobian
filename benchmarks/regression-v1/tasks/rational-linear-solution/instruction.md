# Exact rational linear solution

Solve the exact linear system in `input.json` over the rationals. Return one
canonical rational string for each declared variable, and show substitution
into every equation in `evidence/answer.txt`, include its SHA-256 digest, and
write `submission.json` to the exact agent-visible `submission_schema.json`.
The task does not expose an independent uniqueness certificate, so claim at
most `COMPUTED`.
