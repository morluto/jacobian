# Elementwise fixed vectors without a global invariant

Hard provisional Regression benchmark derived from DeepTheorem row 10001.

## Selection and shortcut audit

The task adds exact finite-group closure plus fixed-space intersection, a
workflow not represented by existing isolated matrix counterexamples. A tiny
single-matrix witness cannot pass: the verifier requires a nontrivial generated
group, a fixed vector for every element, and a zero common fixed space. The
public source gives one construction, but submissions may use other odd fields,
conjugate actions, generator pairs, and per-element vectors; the checker never
matches the published answer.

Nearby DeepTheorem rows about routine set partitions, elementary integrals, and
answer-visible algebra were rejected for insufficient diagnostic depth.

## Difficulty and assurance

Difficulty is Hard (provisional): success requires coordinating group closure,
determinant-one constraints, individual nullspaces, and a global kernel
intersection. Weaker agents are expected to check only the generators or reuse
one vector globally; stronger agents should produce a complete certificate.
All finite-field calculations are independently recomputed, but the result is
`COMPUTED`, not a classification theorem or proof-assistant verification.
