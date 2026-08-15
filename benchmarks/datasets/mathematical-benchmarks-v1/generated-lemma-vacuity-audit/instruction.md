# Audit two generated proof lemmas

The frozen input contains two generated intermediate lemmas attached to IMO
problems. Determine whether each lemma makes non-vacuous progress toward its
stated intent.

For the square-bound lemma:

1. give an in-range pair of distinct cards and an existential witness that
   satisfies the frozen implication only because its antecedent is false;
2. reconstruct the intended universal square-witness contract using the
   supplied logical-AST vocabulary; and
3. give a separate in-range square-sum instance on which that corrected
   contract has a true antecedent and valid bounds.

For the common-divisor lemma, give positive unequal integers and `d = 1` that
satisfy all three divisibility conclusions while the original theorem premise
is false.

Write `submission.json` to the exact agent-visible schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
