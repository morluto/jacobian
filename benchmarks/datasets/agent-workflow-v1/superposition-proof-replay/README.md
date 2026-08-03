# Binary first-order resolution proof replay

The stable task ID remains `jacobian/superposition-proof-replay`, but this v1
benchmark implements binary first-order resolution only. It does not claim to
implement the full superposition calculus.

Reconstruct and independently replay a four-step first-order resolution proof
from a shuffled TPTP-derived clause set.

The task is derived from `reasoning-core/tptp_math_reasoning`, test row 1 at
revision `e87c09a3ca068305c2a869cde3edec5570f08e42` (CC BY 4.0). The public
dependency answer is not trusted: the hidden verifier parses terms, standardizes
parent variables apart, unifies complementary literals, constructs every binary
resolvent, and compares clauses modulo variable renaming, literal order, and
equality orientation.

## Curation

This case was selected because it adds independently checked first-order proof
reconstruction rather than another fixed proof DAG or numerical certificate.
Nearby one-step reconstructions, binary entailment labels, and premise-selection
rows were rejected because answer matching or shallow resolution would dominate.

The benchmark family is **Regression** and the single primary objective is
resolution-proof reconstruction. Difficulty is provisionally **Hard**: a weaker
agent may infer parent edges from superficial overlap but fail variable
standardization or complementary-literal unification; a stronger agent should
produce a valid topological derivation; a tool-less agent can reason manually but
must track several substitutions. Empirical calibration is not yet available.

## Shortcut and assurance boundary

The verifier does not compare against the dataset answer and accepts any valid
topological derivation using all eight frozen clauses exactly once as either an
axiom or derived node. Publishing the source makes memorization possible, so this
is not held-out evidence. The verifier certifies only replay within the frozen
resolution calculus; it does not establish the semantic correctness of the
upstream TPTP ontology or authorize `VERIFIED`.
