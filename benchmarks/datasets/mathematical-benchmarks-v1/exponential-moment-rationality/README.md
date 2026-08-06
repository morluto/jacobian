# Exponential-moment rationality certificate

This Regression benchmark is derived from FineProofs-SFT row 356 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0). It asks for branch-complete rational formulas recovering the fourth two-atom exponential moment from the first four moments.

The verifier does not match the published formulas. It parses arbitrary bounded sparse numerator/denominator polynomials, substitutes the symbolic two-atom moment model, and checks the generic and rank-one branches exactly. The singular branch must be genuinely usable where the generic denominator vanishes. Difficulty is **Hard (provisional)** pending baseline calibration.
