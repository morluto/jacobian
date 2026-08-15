# Audit the squarefree-class argument

Prove the frozen universal claim by connecting three layers rather than by constructing one example set:

1. classify positive integers by their squarefree kernel and establish exactly when a product is a square;
2. translate the ordered-pair count into a sum of squares of class sizes and an independent transversal into distinct classes;
3. give a complete modular certificate showing that `2023` cannot be a sum of at most three integer squares.

You may choose any modulus within the frozen bounds. Submit its complete set of quadratic residues and the exact target residue; residue order is not scored. The verifier will independently enumerate all zero-, one-, two-, and three-square residue sums; checking only selected decompositions is insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
