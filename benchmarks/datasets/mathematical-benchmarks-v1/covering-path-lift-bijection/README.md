# jacobian/covering-path-lift-bijection

Constructs a fiber bijection by lifting a freely chosen base path through a
finite connected graph cover and replaying the reversed path as its inverse.

## Benchmark classification

- **Family:** Regression
- **Primary objective:** constructive unique-path-lifting reasoning
- **Difficulty:** Hard (provisional; no empirical baseline yet). The agent must
  validate a local covering, choose a nontrivial path, propagate every lift,
  and establish the inverse relationship rather than compare cardinalities.
- **Quality score:** 89/100

## Provenance

- Dataset: `Jiahao004/DeepTheorem`
- Revision: `f5935720f176cedff4ecd8ebf83d1696e31cfac8`
- Split/row/source ID: `train` / `0` / `87`
- Canonical source-row digest: `sha256:a0b2d9c381ca5ac7596b5940810c19b09be1cbe8f3cc1d123e0eba398f2a00c2`
- License: MIT

The StackExchange-derived source asks why connected coverings have constant
fiber cardinality. This public regression freezes a finite graph-cover
realization of the path-lifting argument; it is not held-out evidence.

## Shortcut and discrimination audit

Reporting that both fibers have three vertices earns no credit. The verifier
requires all local stars to satisfy the covering condition, every submitted
trace to be the unique lift of one freely chosen valid path, all endpoints to
form a bijection, and reversed-path lifting to recover every source. Different
valid base paths and their induced bijections are accepted.

Weaker agents are expected to count fibers, omit uniqueness, or fail to bind
the reverse lifts to the forward endpoints. Stronger agents should recognize
the constructive inverse argument. Tool-less agents can solve it, but must
track a complete multi-sheet lift certificate exactly.

## Portfolio contribution and verifier boundary

This adds topological transport through a finite cover, distinct from graph
invariant computation, graph counterexamples, fixed proof-DAG replay, and
matrix change-of-basis tasks. The clean-room verifier checks only the frozen
finite graph cover and does not certify the unrestricted topological theorem;
assurance is capped at `COMPUTED`.
