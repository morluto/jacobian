# Recover the next exponential moment

For real `x,y`, define `A=e^x+e^y`, `B=xe^x+ye^y`, `C=x^2e^x+y^2e^y`, `D=x^3e^x+y^3e^y`, and `E=x^4e^x+y^4e^y`.

Submit two rational formulas for `E` using only `A,B,C,D`:

1. a generic formula whose denominator is nonzero off the rank-one locus and vanishes when `x=y`;
2. a singular-branch formula that remains usable when `x=y`.

Each numerator and denominator is a canonical sorted sparse polynomial over variables `[A,B,C,D]` with rational coefficients and total degree at most 4. The verifier accepts any formulas satisfying the symbolic contracts, not only the source proof’s presentation.

Explain the branch split and why rational `A,B,C,D` make `E` rational. Do not claim `VERIFIED`; the checker performs exact symbolic computation without a proof assistant.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `BRANCH_COMPLETE_FORMULAS_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
