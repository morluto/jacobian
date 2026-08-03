# Erdos-Schur f(4) determination

Determine `f(4)` and submit the four color classes of an explicit sum-free
partition of `{1,...,44}`. Each class must be sorted and together the classes
must contain every integer exactly once. Set `upper_bound_method` to
`INDEPENDENT_EXHAUSTIVE_CSP`; the verifier independently checks that no
sum-free 4-coloring of `{1,...,45}` exists. Bind exactly one
`evidence/answer.txt` artifact and claim at most `COMPUTED` assurance.

Let f(k) be the least N such that every k-coloring of {1,...,N} contains a monochromatic solution to a+b=c. Determine f(4) and justify both bounds exactly: a k-coloring of {1,...,N-1} with no monochromatic a+b=c (lower bound) and evidence that no such coloring exists for N (upper bound). Write `submission.json` to the exact agent-visible `submission_schema.json`, record the justification in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence list. Claim `COMPUTED` assurance; do not claim `VERIFIED`.
