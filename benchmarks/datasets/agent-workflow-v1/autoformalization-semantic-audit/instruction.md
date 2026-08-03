# Audit an autoformalized orthogonality statement

Compare the frozen informal theorem, its reference formalization, and a proposed
Lean formalization. Determine whether the proposal preserves the dimension
premise and the meaning of the dot product.

Supply exact integer-vector certificates for both semantic defects:

1. a one-dimensional instance showing why omitting `k ≥ 2` changes the claim;
2. a two-dimensional instance with a nonzero vector orthogonal in the genuine
   dot-product sense, but not coordinatewise annihilated.

Do not claim that either Lean declaration compiles: no Lean runtime is part of
this task. Write `submission.json` to the exact agent-visible
`submission_schema.json`. Put a concise audit in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and bind that file
with its SHA-256 digest.
