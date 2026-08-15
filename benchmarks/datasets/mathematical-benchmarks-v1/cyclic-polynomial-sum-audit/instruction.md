# Audit a cyclic polynomial-system claim

The frozen input gives a cyclic system over the complex numbers, a pairwise-distinctness condition, and a proposed pair of possible values for `s = a+b+c`. Audit that proposal exactly.

Submit `/app/submission.json` following `/app/submission_schema.json`, plus a
digest-bound JSON envelope at
`/app/evidence/cyclic-elimination-certificate.json`. The envelope must contain
with the latter three matching the submission.

Your result must contain:

- the primitive, square-free integer polynomial of least degree that your exact elimination shows is necessary for `s` under the system and the pairwise-distinctness condition, with coefficients in descending degree order and positive leading coefficient;
- the exact reduced rational value of that polynomial at each proposed sum, in the same order as the frozen input;
- the resulting classification of each proposed value as `PASSES_NECESSARY_CONDITION` or `FAILS_NECESSARY_CONDITION`;
- all remaining real roots of the necessary polynomial after any root excluded by pairwise distinctness is removed, represented as quadratic irrational objects `{"rational":"p/q","radical_coefficient":"r/t","radicand":d}` in increasing order;
- the rational candidate excluded by the original system, together with the exact elementary-symmetric invariants on that branch and the nonzero residual in the product consequence
  `(a^2-6)(b^2-6)(c^2-6)=abc`.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/cyclic-elimination-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
