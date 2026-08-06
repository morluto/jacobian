# Polynomial divisibility uniqueness

## Evaluation contract

- **Benchmark family:** Regression
- **Primary reasoning objective:** symbolic divisibility and uniqueness reasoning
- **Provisional difficulty:** Hard. The task requires a length-13 symbolic reduction, a polynomial-gcd completeness argument, and a separate exact quotient identity. Expected weak-model failures include testing only the public parameter or omitting uniqueness; stronger agents should produce the whole certificate. Tool-less agents can derive the recurrence but face substantial bookkeeping.
- **Assurance boundary:** the clean-room verifier establishes exact polynomial identities over `Q[a]` and `Z[x]` and reports `COMPUTED`; no proof assistant certifies the prose proof.

## Provenance

Derived from `ChristianZ97/PutnamBench-lean4`, revision `f753021f5fd7e40da05e137c0d0cb8624790f227`, default/train row 15 (`putnam_1963_b1`), canonical row SHA-256 `074675c531ddfa931d6fedffb262f603292fe5fa67e95d7a5f60a6de45b4c017`. License: Apache-2.0.

## Shortcut audit

The public answer `a=2` is visible and therefore cannot earn credit alone. The submission must expose both full symbolic remainder families, their independently recomputed monic gcd, and an exact quotient whose product reconstructs the degree-13 dividend. Tiny-witness checking, bounded search, answer-pattern recognition, forged assurance, malformed arrays, and evidence-path substitution all fail. The verifier does not execute submitted code or trust the public solution.

## Portfolio contribution

This is the first task whose completeness boundary is a symbolic parameter gcd for a divisibility family. It differs from polynomial normalization, fixed root reconstruction, bounded modular obstruction, and candidate-only factor checking.
