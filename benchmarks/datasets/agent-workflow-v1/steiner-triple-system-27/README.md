# Steiner triple system of order 27

This Regression benchmark is a closed finite slice of CombiBench
`brualdi_ch10_34` (test row 38), pinned to revision
`ded8b32e2a5ac01cac6b2b89d0b1edc43215cff4` under the MIT license.

## Selection and portfolio value

It adds nontrivial combinatorial-design construction with exhaustive pair
incidence verification. Routine counting rows and graph facts were rejected as
low-discrimination or workflow duplicates. Unlike a fixed-answer task, this
accepts every labeled Steiner triple system on the declared point set.

## Family, objective, difficulty, and shortcuts

- Family: Regression.
- Primary objective: construct a complete finite incidence design.
- Difficulty: Hard (provisional). A valid response must coordinate 117 blocks
  and 351 exact pair obligations; one local error invalidates completeness.

A memorized scalar answer, tiny witness, partial design, or repeated block
cannot pass. The public affine construction is a possible strategy, but the
verifier does not prescribe or recognize it. Weaker agents are expected to
produce omissions or duplicate pairs; stronger code-capable agents should
construct and audit the full design. Tool-less success remains possible but
requires substantial bookkeeping.

## Assurance boundary

The clean-room verifier exhaustively checks the finite incidence contract and
accepts alternative designs and point relabelings. It does not prove the general
`v^t` theorem or replay Lean, so assurance remains `COMPUTED`.
