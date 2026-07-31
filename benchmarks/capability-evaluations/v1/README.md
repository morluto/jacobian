# Capability evaluation v1

This directory freezes the discovery handoff and comparison design derived
from the public 24-task Harbor `regression-v1` suite. It does not contain a
model result or a held-out score.

The artifacts have three distinct roles:

- `gap-ledger.json` classifies the mathematical moves and portfolio gaps
  exposed by every public task. Its accepted candidates are discovery
  handoffs, not performance claims.
- `comparison-plan.json` fixes the C0/C1/C2 meanings, matched-condition
  invariants, contamination boundary, and execution gate.
- `job-public-c1-current.json` and `job-public-c2-treatment.json` are Harbor
  configurations for answer-visible workflow reproduction only. They differ
  only in their result directory and digest-pinned Jacobian image selection.

The public jobs must never be cited as causal evidence. C1 is the frozen
current catalog; C2 is the same catalog plus exactly one candidate delta. A
future held-out run must supply a separately frozen task set whose Oracle
material is stored outside the evaluated workspace, use at least two
independent mathematical families, and bind every result to the exact
catalog, policy, image, model, prompt, budget, seed, scorer, and task digests.

No held-out fixtures or Oracle answers are committed here. Until those
external identities are filled and a hard model-run budget is authorized,
`comparison-plan.json` remains `SCAFFOLD_ONLY` and no model execution is
allowed.

