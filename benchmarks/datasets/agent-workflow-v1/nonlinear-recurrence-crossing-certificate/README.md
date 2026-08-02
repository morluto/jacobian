# jacobian/nonlinear-recurrence-crossing-certificate

Certify that the recurrence `a₁=56`, `aₙ₊₁=aₙ-1/aₙ` has a negative term
before index 2002 using a two-phase potential argument.

This Regression benchmark is derived from `lm-provers/FineProofs-SFT` train row
352 at revision `73661e62811cf2940a0d3f82788a4f4332204c2f` (Apache-2.0).
The verifier independently checks the Laurent-polynomial identity, reduced
rational threshold and decrement, minimal phase budget, and an unordered set
of typed rational interval images. It does not accept fixed prose labels for
the terminal argument.

Quality score: **86/100** after structured-certificate review. The single objective is nonlinear recurrence
crossing certification. Difficulty is **Hard (provisional)**: the solver must
invent a squared-potential phase and a short terminal argument. Brute-forcing
floating-point iterates cannot pass the exact rational certificate. Elementary
order-preservation lemmas are trusted, so assurance is capped at `COMPUTED`.
