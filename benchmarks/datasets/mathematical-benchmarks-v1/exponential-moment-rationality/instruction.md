# Recover the next exponential moment

For real `x,y`, define `A=e^x+e^y`, `B=xe^x+ye^y`, `C=x^2e^x+y^2e^y`, `D=x^3e^x+y^3e^y`, and `E=x^4e^x+y^4e^y`.

Submit two rational formulas for `E` using only `A,B,C,D`:

1. a generic formula whose denominator is nonzero off the rank-one locus and vanishes when `x=y`;
2. a singular-branch formula that remains usable when `x=y`.

Each numerator and denominator is a canonical sorted sparse polynomial over variables `[A,B,C,D]` with rational coefficients and total degree at most 4. The verifier accepts any formulas satisfying the symbolic contracts, not only the source proof’s presentation.

Report the branch split and rationality conclusion in the typed result. The checker performs exact symbolic computation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
