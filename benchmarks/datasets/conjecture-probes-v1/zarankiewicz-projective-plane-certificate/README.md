# Zarankiewicz projective-plane certificate

This public Regression benchmark turns retained conjecture row C-044 into a
closed finite task. The agent must construct the order-three projective-plane
incidence graph and an exact extremal certificate for
`z(13,13;2,2)=52`.

The primary reasoning objective is finite extremal-certificate construction.
The verifier does not trust a named construction: it canonicalizes projective
classes, recomputes incidence over `F3`, exhaustively rejects every `K2,2`, and
checks the pair-count upper bound excluding 53 edges.

## Curation and difficulty

Provisional difficulty is Hard: the response coordinates finite-field
normalization, a complete 52-edge incidence relation, two-sided pair
intersection counts, and the convexity/pair-budget upper bound. Weaker agents
are expected to omit projective representatives or confuse affine and
projective incidence; stronger agents should produce a replayable certificate.

The shortcut audit rejects tiny planes, copied edge counts without coordinates,
partial incidence tables, and label-only extremality claims. Although the
projective-plane construction is public, the complete typed certificate is
independently recomputed rather than answer-matched. This task is distinct from
existing coloring, minor, homology, and generic incidence-determinant tasks.

## Provenance and boundary

- Inventory source: retained `Unresolved Conjectures` row C-044 (Zarankiewicz).
- Mathematical source: Kővári, Sós, and Turán, *On a problem of K. Zarankiewicz*,
  Colloquium Math. 3 (1954), 50–57.
- Construction source: the standard point-line incidence structure of
  `PG(2,3)`; all finite claims are reconstructed locally.

Full reward means `CHECKED` for this one finite graph and its exact finite
upper bound. It does not prove any unresolved asymptotic or general
Zarankiewicz statement.
