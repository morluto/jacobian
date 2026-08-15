# Balanced row-permutation construction

The frozen `8 x 6` matrix uses symbols `1..4`, each occurring exactly twelve
times.  Permute entries **within each row only** so that every output column
contains exactly two copies of each symbol.

Submit the six column layers in order.  Each layer must assign every row to one
unused input position, record the corresponding symbol, and exhibit the
required two-of-each-symbol balance.  Also submit the resulting matrix and the
eight row permutations, one for each input row.

The verifier independently checks the global occurrence premise, every
position binding, exactly-once use of all input positions, all row
permutations, and all column histograms.  Alternative valid decompositions are
accepted.  Place the structured answer at `/app/submission.json` and a bound
explanation at `/app/evidence/answer.txt`.  The evidence must contain exactly
one `RESULT_JSON:` line whose JSON equals the submitted `result` object, and
must discuss column layers, row permutations, source positions, and the exact
two-of-each-symbol balance.

Do not claim proof-assistant verification; the result is an exact computation
for this frozen instance.  Claim at most `COMPUTED` assurance.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
