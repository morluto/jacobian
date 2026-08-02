# Continuant reversal certificate

This Regression benchmark is derived from FineProofs-SFT row 302 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0). It asks for a symbolic path-tiling interpretation of a variable-coefficient recurrence and the reflection involution proving reversal symmetry.

The verifier independently enumerates all admissible monomial supports at `n=10` and checks the complete reflection pairing. This finite replay validates the symbolic certificate shape; the arbitrary-`n` theorem still relies on the submitted general tiling recurrence and reflection rule. Difficulty is **Hard (provisional)** pending baseline calibration.
