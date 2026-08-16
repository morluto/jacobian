# Classify real-rooted sign polynomials

Classify every nonconstant polynomial whose coefficients all belong to
`{-1,1}` and whose complex roots are all real, including overall sign
changes.

Submit a universal degree-bound certificate based on monic normalization,
Newton's second power-sum identity, the root product, and AM-GM. Then submit a
complete audit of every sign-coefficient polynomial in the remaining degrees
1 through 3. Coefficients are in ascending degree order. For quadratics and
cubics, include the exact discriminant and the resulting real-rooted decision.
The verifier independently enumerates the whole finite residue class and
recomputes every discriminant; a copied final list is insufficient.

`result` (the same result object placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
