# Permutation inversion involution

This Hard (provisional) Regression benchmark turns CombiBench row 20 into executable combinatorial reasoning: the agent must select and parameterize a transformation, provide non-fixed traces, and derive a global inversion total. The verifier independently enumerates all 5,040 permutations, replays the transformation, checks involution and fixed-point properties, and recomputes every inversion count.

It adds algorithm selection plus exhaustive semantic replay, rather than another proof audit or fixed proof-label certificate. The public hint exposes that a pairing exists, but not which offered transformation works; answer-only and trace-only shortcuts fail. Assurance is `COMPUTED` for the frozen `n=7` instance and does not prove the theorem for arbitrary `n`.

Quality score: 86/100. Source: AI-MO/CombiBench row 20, revision `882ba08befd0856f5364db1e53d58c7e2cf704f9`, MIT license.
