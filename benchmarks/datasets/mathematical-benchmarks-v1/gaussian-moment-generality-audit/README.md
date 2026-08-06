# Gaussian-moment generality audit

This Hard benchmark turns a conversation-prompted research construction into an offline proof-generality audit. It tests whether an agent can distinguish finite experimental evidence from an all-exponents proof and then repair the gap with a replayable formal rational-function certificate.

The source is Christopher D. Long, *Small Counterexamples to the Gaussian Moments Conjecture*, arXiv:2607.18186v1. The task uses a paraphrased mathematical template, pins the source version and archive digest, and does not reproduce the paper text.

The verifier accepts a bounded rational one-parameter family. It independently reconstructs `h`, evaluates `v` and `zeta` at the inverse branch, and checks the branch equation, correction-factor identity, and both generating functions as exact rational-function identities. The quadratic `v` is validated through the correction factor it satisfies, not by pinning it to a closed-form expression. This is not a finite moment table and not a fixed-answer string check.

Assurance is capped at `COMPUTED`. The frozen Lagrange/Gaussian identity is an explicit premise of the task; no proof assistant checks that premise, the full paper, or the open two-dimensional case.
