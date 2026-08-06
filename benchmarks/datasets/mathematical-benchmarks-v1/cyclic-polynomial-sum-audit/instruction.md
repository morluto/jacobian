# Audit a cyclic polynomial-system claim

The frozen input gives a cyclic system over the complex numbers, a pairwise-distinctness condition, and a proposed pair of possible values for `s = a+b+c`. Audit that proposal exactly.

Submit `/app/submission.json` following `/app/submission_schema.json`, plus digest-bound prose at `/app/evidence/answer.txt`.

Your result must contain:

- the primitive, square-free integer polynomial of least degree that your exact elimination shows is necessary for `s` under the system and the pairwise-distinctness condition, with coefficients in descending degree order and positive leading coefficient;
- the exact reduced rational value of that polynomial at each proposed sum, in the same order as the frozen input;
- the resulting classification of each proposed value as `PASSES_NECESSARY_CONDITION` or `FAILS_NECESSARY_CONDITION`;
- all remaining real roots of the necessary polynomial after any root excluded by pairwise distinctness is removed, represented as quadratic irrational objects `{"rational":"p/q","radical_coefficient":"r/t","radicand":d}` in increasing order;
- the rational candidate excluded by the original system, together with the exact elementary-symmetric invariants on that branch and the nonzero residual in the product consequence
  `(a^2-6)(b^2-6)(c^2-6)=abc`.

The evidence prose must explain the derivation in your own words, including how pairwise distinctness is used. It must not merely repeat the JSON fields.

Claim `COMPUTED` assurance and complete scope exactly as specified by the schema. This benchmark checks an exact algebraic audit of the frozen system; it does not machine-check the original contest solution or provide proof-assistant verification.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently replays the exact elimination, proposed-value evaluations, radical roots, and excluded-branch residual.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `CLAIM_INCONSISTENT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
