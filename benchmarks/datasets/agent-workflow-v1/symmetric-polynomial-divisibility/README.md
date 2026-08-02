# Symmetric-polynomial divisibility certificate

This Regression benchmark is derived from FineProofs-SFT row 323 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0). It asks for an exact sparse-polynomial ideal-membership certificate proving a divisibility implication without testing numerical instances.

The verifier independently multiplies the submitted rational multiplier polynomials by the two hypothesis generators and requires their sum to equal the target polynomial coefficient by coefficient. Multiple valid certificates are accepted. The certificate establishes the algebraic identity over `QQ`; the final modular implication uses only that integer multiples of `n` are closed under integer linear combinations. Difficulty is **Hard (provisional)** pending baseline calibration.
