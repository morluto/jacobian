# Factor and certify reciprocal symmetry

The frozen input gives a degree-16 integer polynomial by coefficients in constant-to-leading order. Recover a complete factorization

`leading_coefficient * product(Phi_order(x)^multiplicity)`

using cyclotomic orders from 2 through 30. Submit the ordered factor list, the independently expanded coefficient vector, its reciprocal vector, `P(1)`, and the reciprocal scalar. Explain why excluding `Phi_1=x-1` corresponds to `P(1) != 0` and why inversion of root-of-unity orbits produces coefficient symmetry.

The verifier reconstructs cyclotomic polynomials and the entire product; coefficient-pattern recognition alone is insufficient. Do not claim `VERIFIED` because no proof assistant certifies the unrestricted source theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
