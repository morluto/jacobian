# Almost-isosceles Pythagorean generator recurrence

This Regression benchmark freezes NaturalProofs train row 54 (ProofWiki
`[13535,0]`) at revision `bdf412319e160046fd966f7d72d776d3d7b866a0`.
It asks for an exact recurrence certificate showing that the generator map
`(m,n) -> (2m+n,m)` preserves the almost-isosceles Pythagorean condition by
negating its quadratic invariant.

The seed is not fixed. The verifier accepts any bounded primitive,
opposite-parity seed with `|m^2-2mn-n^2|=1`, then independently rebuilds eight
generators and all three sides. It checks the matrix action, determinant,
quadratic sign flip, coprimality, parity, Pythagorean identity, and unit leg
gap. Assurance is `COMPUTED`, not proof-assistant verification.

## Curation

The task adds symbolic invariant discovery plus an executable recurrence trace.
Nearby elementary group and set-theory rows were rejected as proof-label or
single-step workflows. The public theorem statement is insufficient because a
candidate must provide a consistent alternative seed and eight exact stages.

Quality score: **86/100**. Difficulty: **Hard (provisional)** because success
requires linking a matrix recurrence, an indefinite quadratic form, primitive
generator conditions, and exact triangle identities. The shortcut audit found
no tiny-witness dominance: one valid triple alone earns no credit.
