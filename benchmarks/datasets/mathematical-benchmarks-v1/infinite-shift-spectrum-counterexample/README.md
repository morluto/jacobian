# Infinite-shift eigenvalue-set scope audit

This Assurance benchmark is derived from ProofNet test row 65 (`Axler|exercise_5_11`) at dataset revision `cb8e75614830035a37f3a2a11de5e625eaf0bc31` (MIT). The proposed formal statement asserts equality of the eigenvalue sets of `ST` and `TS` without a finite-dimensional hypothesis.

The agent must construct unilateral left/right shifts on the vector space of finitely supported rational sequences, replay both compositions on an exact basis window, and isolate the zero-eigenvalue asymmetry. The verifier derives every action from the declared symbolic shift rules and accepts either assignment of the two shifts. The finite window checks consistency with those rules; the general composition identities follow from the rules themselves rather than from finite enumeration. “Eigenvalue set” here is the point spectrum, not the full operator spectrum. Difficulty is **Hard (provisional)** pending empirical calibration.

The assurance ceiling is `COMPUTED`: the benchmark checks the frozen algebraic operator model and its declared general shift rules, not Lean elaboration, the complete operator spectrum, or a machine-checked theorem.
