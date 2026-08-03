# Real-rooted sign-polynomial classification

This Regression benchmark is derived from PutnamBench
`putnam_1968_a6` at immutable revision
`dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c` (Apache-2.0 Lean source;
informal statement distributed with MAA permission). The public list is not
trusted. The verifier derives the finite search boundary from the submitted
universal certificate, enumerates all 28 remaining sign polynomials, and
recomputes exact quadratic/cubic discriminants.

## Quality and shortcut audit

Quality score: **87/100**. Primary objective: complete symbolic
classification. Difficulty is **Hard (provisional)** because success requires
combining a universal root-moment argument with exhaustive exact algebra and
completeness accounting; empirical calibration is pending. The public list of
twelve polynomials cannot pass without the full 28-case audit and consistent
degree-bound certificate. The task is not a tiny witness or numerical root
sampling problem.

The verifier proves the finite discriminant audit exactly. It trusts Newton
identities, Viète, AM-GM on squared real roots, and the standard discriminant
criteria, so assurance is capped at `COMPUTED`.
