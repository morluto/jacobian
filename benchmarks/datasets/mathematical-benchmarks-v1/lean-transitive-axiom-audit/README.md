# Lean transitive-axiom audit

This task freezes the minimal reproductions from Lean issue #8840. It asks for
a trust audit of shallow axiom reports rather than a theorem proof.

The clean-room verifier computes dependency closure over the frozen declaration
graph and checks which dependencies are absent from each observed report. It
does not execute Lean or claim that the defect persists in current Lean.
