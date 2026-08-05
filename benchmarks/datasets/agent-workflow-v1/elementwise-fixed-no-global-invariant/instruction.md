# Elementwise fixed vectors without a global invariant

The offline input freezes a claim that swaps two quantifiers for finite linear
actions. Refute it by constructing a subgroup of `SL_3(F_q)` for one allowed
odd prime.

Submit two generators, the complete generated group in lexicographic matrix
order, and one nonzero fixed vector for every listed group element. The
verifier independently closes the generators under multiplication, checks
determinants, replays every fixed-vector equation, and computes the common
fixed-space intersection. The group must have order between 6 and 48, and its
common fixed space must be zero.

This is not a request to reproduce the public example. Alternative generators,
fields, conjugates, and fixed vectors are accepted whenever they satisfy the
contract. Explain the quantifier failure in `evidence/answer.txt` and bind that
file by SHA-256.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently generates the finite group, checks determinant one and every elementwise fixed vector, and computes the common fixed-space intersection.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `ELEMENTWISE_DOES_NOT_IMPLY_GLOBAL`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
