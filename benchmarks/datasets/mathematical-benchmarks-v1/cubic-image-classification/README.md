# Cubic-form image classification

This Regression-family task freezes Putnam 2019 A1 from PutnamBench's Apache-2.0
Lean source at commit `dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c`.

Its single objective is complete symbolic image classification: combine a
factorization and exhaustive modular obstruction with parametric constructions
covering every allowed residue class.  Difficulty is **Hard (provisional)**:
weaker agents are expected to stop at the public residue obstruction, while
stronger agents must synthesize and validate a complete family cover.

Shortcut audit: neither the public answer, finite residue enumeration, nor a
tiny witness passes without three symbolic construction families and domain
coverage.  The schema does not reveal their coefficients.  The verifier uses
clean-room integer polynomial arithmetic and reports `COMPUTED`, not a
proof-assistant theorem.
