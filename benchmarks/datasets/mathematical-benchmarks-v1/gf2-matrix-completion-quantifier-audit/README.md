# GF(2) matrix-completion quantifier audit

Hard-provisional Regression benchmark derived from Xerv-AI/GRAD train row 6.
The source confuses an existential minimization over completions with the
availability of a freely chosen full-rank completion, and incorrectly infers
symmetry from band support.

The agent must submit an asymmetric band-supported pattern and two completions
respecting exactly the same forced-one constraints: one of rank one and one of
full rank. The verifier independently performs Gaussian elimination over
GF(2). Quality score: **86/100**.
