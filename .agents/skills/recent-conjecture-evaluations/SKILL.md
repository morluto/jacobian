---
name: recent-conjecture-evaluations
description: Evaluate Jacobian reliability using recently resolved conjectures as held-out probes.
---

# Recent Conjecture Evaluations

Use recently resolved conjectures to probe Jacobian reliability. The outcome is
an evidence-backed diagnosis, not a collection of solved examples. This skill
owns source selection, deterministic replay, optional model comparisons, and
attribution; Harbor packaging is a separate workflow.

## Choose the requested mode

- **Probe:** read [probe workflow](references/probe.md) for one source cycle,
  including source gating, independent oracle, current-main replay, and reporting.
- **Review:** inspect the completed cycle against
  [source gating](references/source-gating.md), the oracle and frozen inputs,
  and the applicable [scoring rules](references/evaluation-and-scoring.md).
  Use [report fields](references/report-schema.md) to identify missing evidence.
  Reproduce disputed deterministic claims where useful; do not repeat model arms.
- **Coordinate:** search saved reports, reservations, and issue/PR ownership for
  the source and root mechanism. Use the inventory helper shown below and the
  ownership rules in [action policy](references/action-policy.md). Do not begin
  a new probe merely to answer a coordination question.

```sh
python .agents/skills/recent-conjecture-evaluations/scripts/search_inventory.py \
  "source or root-cause phrase" outputs benchmarks/results
```

## Preserve the evidence boundary

Bind prompts, payloads, and oracles to the exact intended input. Reconstruct gold
independently of evaluated model arms and distinguish source-supplied replay from
an independent oracle. Record precise source status and dates. Computation,
verification, imported theorems, and unproved claims establish different things.
Timeouts, unavailable operations, and missing witnesses are non-conclusions.
Never attribute a model's fallback or transcription error to Jacobian.

Audit new probes deterministically on current main before paid model calls.
Run comparisons only when they resolve uncertainty that direct evidence cannot,
with user-authorized cost boundaries and frozen control/treatment conditions.
Do not weaken verification or create a benchmark-specific operation to make a
probe pass. Before external action, apply the
[action policy](references/action-policy.md) within existing user authorization.
This evaluation workflow produces at most a localized draft PR; merging is a
separate user-authorized task.

## Completion and stalls

Complete one requested cycle by default, including rejected and deterministic-only
outcomes. An oracle disagreement stops dependent model evaluation while allowing
bounded diagnosis. A known unavailable operation skips dependent work, not the
attribution and report. Preserve checkpoints and raw evidence at phase boundaries;
inspect a phase without a checkpoint for 30 minutes.

Additional cycles or model calls must fit the operator's requested scope and
cost limits. Retry a cancelled call only with new diagnostic evidence. Stop
collecting examples for an owned root cause once the agreed independence
threshold is met. Report unresolved evidence and a distinct next direction
without silently starting it.
