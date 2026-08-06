# Multiplicative-grid extremum

This Hard (provisional) Regression benchmark asks for a 100-element set of
positive integers with the maximum possible number of pairs whose ratio is 2
or 3. The agent must provide both an exact extremal witness and the projection
data behind a universal upper bound.

The verifier independently factors every submitted integer as
`core * 2^a * 3^b`, recomputes every good pair, checks the lattice projections,
and derives the universal bound with a finite dynamic program. Alternative
valid witnesses, including common rescalings coprime to 6, are accepted. The
elementary projection lemma is part of the trusted mathematical boundary, and
no proof assistant is invoked, so assurance is capped at `COMPUTED`.

## Selection rationale

The source row was selected because it supports both a nontrivial construction
and an independently replayable universal optimality certificate. Nearby
CombiBench candidates were rejected when their general proof depended on an
unformalized game strategy, duplicated an existing permutation workflow, or
lacked a compact independently checkable certificate.

## Shortcut audit

Knowing the published answer `180` is insufficient. A successful submission
must supply 100 distinct integers, the complete 180-edge relation, exact
factorizations, and projection data that the verifier recomputes. Tiny-witness,
answer-only, sampling, and evidence-rebinding shortcuts are rejected.
