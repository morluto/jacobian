# Cyclic polynomial sum audit

This Regression benchmark is derived from `INSAIT-Institute/BrokenMath` benchmark row 31 at immutable revision `5eda8c5fbd150afde41b6206b60700ab7d8e25c7` (CC-BY-NC-SA-4.0).

The source turns a contest problem into an adversarial request to prove an incorrect pair of possible sums for a cyclic quadratic system. The benchmark asks for an exact audit without revealing the elimination polynomial or the repair route. The clean-room verifier independently derives the elimination consequence, evaluates both proposed values, checks the remaining quadratic roots, and verifies why the extraneous rational branch violates a product consequence of the original system.

Family: **Regression**. Primary reasoning objective: **adversarial theorem diagnosis by symbolic elimination**.

Difficulty is **Hard (provisional)**: the key elimination and side-condition interaction are not supplied, and the task requires coordinating exact polynomial algebra with a semantic exclusion. Baseline calibration is pending.

The shortcut audit rejects answer-label filling: every numerical classification is recomputed, the polynomial must be primitive and square-free, and evidence must explain a derivation. The public source is contamination-prone, but its adversarial claimed answer is wrong and cannot pass the verifier.

Assurance is capped at **COMPUTED**. The verifier establishes the exact algebraic consequence for the frozen system; it does not prove the original contest problem in a proof assistant.
