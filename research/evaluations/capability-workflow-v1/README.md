# Capability workflow evaluation v1

This directory contains the non-runnable discovery handoff and comparison
design derived from the public 26-task Harbor `agent-workflow-v1` suite. It is
research metadata, not a Harbor dataset: it contains no executable tasks, job
configuration, Compose file, model result, or held-out score.

The executable public cases remain the canonical
`benchmarks/datasets/agent-workflow-v1` dataset. Reusable Harbor support lives
under `benchmarks/tooling`; this directory may reference those datasets and
tools, but it does not copy their task lists or inject research records into
agent containers.

The artifacts have three distinct roles:

- `gap-ledger.json` classifies the mathematical moves and portfolio gaps
  exposed by every public task. Its accepted candidates are discovery
  handoffs, not performance claims.
- `comparison-plan.json` fixes the C0/C1/C2 meanings, matched-condition
  invariants, contamination boundary, and execution gate. Its conditions have
  no committed Harbor job while that gate is closed.

The public reproduction is a fixed, answer-visible suite and must never be
cited as causal evidence. C1 is the frozen current catalog; C2 is the same
catalog plus exactly one candidate delta. A future held-out run must supply a
separately frozen task set whose Oracle material is stored outside the
evaluated workspace, use at least two independent mathematical families, and
bind every result to the exact catalog, policy, image, model, prompt, budget,
seed, scorer, and task digests.

No held-out fixtures or Oracle answers are committed here. Until those
external identities are filled and a hard model-run budget is authorized,
`comparison-plan.json` remains `SCAFFOLD_ONLY` and no model execution is
allowed. Plans and handoffs in `research/evaluations/` are not performance
evidence without a separately frozen held-out evaluation.
