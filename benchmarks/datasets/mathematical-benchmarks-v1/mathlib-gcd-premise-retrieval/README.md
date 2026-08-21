# jacobian/mathlib-gcd-premise-retrieval

This workflow benchmark measures whether an agent can retrieve and apply a
Lean declaration that closes a frozen natural-number gcd goal.

The task exposes no operation name or required tool sequence. A Jacobian-enabled
run may discover and use Mathlib declaration search as an experimental
treatment; correctness depends only on whether the submitted declaration
application elaborates in the independent pinned Lean verifier.

The case is derived from the public `lean-retrieval` reproduction, but the
answer remains hidden from the agent at runtime. It is a regression benchmark,
not a held-out theorem-discovery claim.
