# Extremal subset-sum semantic audit

This Assurance-family task is derived from google-deepmind/formal-conjectures
issue 4440 at the immutable source revision recorded in `task.toml`. It asks an
agent to diagnose two independent semantic defects in a Lean formalization of
Erdős Problem 361 and to support both diagnoses with finite exact certificates.

The verifier independently enumerates the relevant finite powersets. It does
not run Lean, assess the corrected asymptotic conjecture, or claim that the
upstream issue report is machine-verified.

Selection rationale: unlike a single typo audit, the task requires separating
binder shadowing from a changed subset predicate, replaying the inconsistent
extremal values, and comparing two distinct finite optimization problems.
Nearby quantifier-only issues were rejected because they substantially overlap
existing scope and quantifier-alignment benchmarks.

Difficulty is **Hard (provisional)**: the mathematical instances are finite,
but success requires source-level binder analysis, predicate semantics,
independent extremal reconstruction, and calibrated assurance. Baseline agent
success has not yet been measured.
