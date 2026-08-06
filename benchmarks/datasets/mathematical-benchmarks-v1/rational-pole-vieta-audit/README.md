# Rational pole and Vieta audit

This **Regression** benchmark is derived from `openai/prm800k`
`phase2_test` row 15 at immutable blob
`32ba2267b92a5ad62ecbe0a454d742f713dd6958` (MIT).

The task isolates a process-supervision failure: a proposed solution confuses
the values under `x^2` with the actual complex poles. The agent must repair the
domain analysis and produce an exact polynomial certificate. The verifier
reconstructs every polynomial by convolution, evaluates the surviving
numerators at all pole squares, and applies Vieta only after confirming that
no cleared-denominator root is extraneous.

Primary reasoning objective: **diagnose and repair pole handling in symbolic
elimination**. Difficulty is **Medium–Hard (provisional)**: the algebra is
bounded, but success requires coordinating rational-function domains,
coefficient arithmetic, and root multiplicity accounting. The shortcut audit
rejects a bare final answer and corrupted coefficient or pole certificates.

Assurance is capped at **COMPUTED**.
