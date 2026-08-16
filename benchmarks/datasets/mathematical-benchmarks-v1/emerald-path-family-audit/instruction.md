# Audit the claimed uniqueness of an APMO path construction

The frozen ProofBench model response concludes that the only possible parameters are `alpha=beta=1`. Audit that conclusion.

Submit a nontrivial positive structured rational pair with `alpha >= beta`, `alpha+beta=2`, and `alpha != beta`. Equivalent encodings such as `6/4` and `3/2` are accepted after exact `Fraction` normalization. Use the path `x_n=ceil(n/2)`, `y_n=floor(n/2)`. Provide an exact parity certificate showing that `floor(x_n alpha+y_n beta)=n` for every nonnegative integer `n`, and a 16-row exact trace.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
