# Euler fourth-power claim scope audit

Refute the frozen universal claim by supplying a primitive positive-integer witness
`x^4 + y^4 + z^4 = w^4` within the declared bound. The three left bases must be
strictly increasing, all four bases must be distinct, and their joint gcd must be one.

Submit the exact fourth powers, both sides of the equality, and residue checks for
every required modulus.

The verifier accepts any witness satisfying this contract; it does not compare
against one memorized tuple. Write `/app/submission.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
