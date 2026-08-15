# Mahler-measure leading-coefficient audit

The frozen evaluator multiplies the moduli of roots outside the unit disk but silently assumes the polynomial is monic. Audit that formula for the supplied nonmonic reciprocal degree-eight polynomial.

Submit:

1. a factorization into exactly four primitive quadratic integer factors, ordered lexicographically by coefficient triple;
2. each factor's exact outside-root contribution in the basis `a+b*sqrt(5)`;
3. the flawed monic-normalized result;
4. the missing leading-coefficient multiplier; and
5. the corrected Mahler measure in the same radical basis.

The verifier multiplies the factors back to the frozen polynomial, classifies the two cyclotomic factors, checks the reciprocal quadratic roots algebraically, and recomputes all radical products. Equivalent factor signs are normalized by requiring positive leading coefficients and primitive coefficient gcd one.

This Assurance result concerns one exact polynomial and one formula defect only. It does not determine Lehmer's problem or compare all integer polynomials. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
