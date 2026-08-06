# Repair and replay a Metamath-style proof

The frozen input contains a small Metamath-style assertion registry, atomic
hypotheses, and a corrupted reverse-Polish proof of the implication
syllogism. Repair the trace with exactly two token replacements and submit the
complete repaired proof.

For every proof token, record the stack depth and top expression after the
token is applied. For assertion tokens, also record the exact variable
substitution inferred from their ordered hypotheses. Atomic tokens have an
empty substitution. Expressions are token arrays and must match the frozen
syntax exactly.

The proof checker pops ordered hypotheses, unifies every pattern variable
consistently, instantiates the conclusion, and requires one final stack item
equal to the target. Merely naming the two repaired labels or asserting the
target is insufficient.

Write `submission.json` to the provided schema. Write `evidence/answer.txt`
with exactly one line beginning `RESULT_JSON:` followed by the exact compact
JSON serialization of `result`. The other text must explain the assertion
applications using the ordered stack and its variable unification/substitution;
unrelated or marker-only text is not evidence. Digest-bind that file and claim
at most `COMPUTED`.
The evidence file must not exceed 16 MiB.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PROOF_REPAIRED_AND_REPLAYED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
