# Putnam 2025 A6 2-adic induction audit

This task freezes Putnam 2025 A6 and its machine-checked Lean solution from
`AxiomMath/putnam2025` at commit
`2653cded72f5112acdc935b4f674711a780af95d`. The source repository is MIT
licensed.

The benchmark does not ask for a finite table of recurrence values. It requires
a symbolic certificate for the simultaneous 2-adic induction used by the
source proof, followed by the valuation transfer from `u_n=2b_n` to the target
difference. The clean-room verifier independently checks the recurrence-
difference identity, the base recurrence values, and the affine valuation
arithmetic for every symbolic `k>=1`, conditional on the two frozen doubling
identities from the cited source. It does not derive those identities from
first principles, run Lean, or independently establish the number-theoretic
facts encoded by the submitted valuation rules.

This adds a universal-induction and assurance-calibration workflow rather than
another bounded recurrence computation.
