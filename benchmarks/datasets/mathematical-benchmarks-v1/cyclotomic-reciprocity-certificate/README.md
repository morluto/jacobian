# Cyclotomic reciprocity certificate

This Regression benchmark is derived from FineProofs-SFT row 358 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0). It asks for a complete cyclotomic factorization certificate connecting roots of unity, the nonvanishing value at one, and reciprocal coefficient symmetry.

The verifier independently constructs every submitted cyclotomic polynomial by exact division of `x^n-1`, multiplies the complete factorization, and checks the expanded and reciprocal coefficient vectors. Merely observing that the public coefficients look palindromic cannot pass. Difficulty is **Hard (provisional)** pending baseline calibration.
