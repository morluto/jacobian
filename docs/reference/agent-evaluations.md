# Agent workflow observations

[Documentation home](../index.md)

Jacobian's fixed workflow observation surface is the Harbor
[`agent-workflow-v1`](../../benchmarks/datasets/agent-workflow-v1/README.md)
dataset. Its self-contained mathematical tasks cover graph, algebra,
linear-algebra, number-theory, geometry, combinatorics, probability, and
formal-mathematics workflows, including the original graph, partition, SAT,
linear-system, Hermite, and polynomial cases.

The task bundles are agent-agnostic. Instructions describe the mathematical
outcome and evidence without naming capability IDs or prescribing
decomposition, verification order, or stopping criteria. Each task freezes its
offline input, Oracle-only solution, and separate clean-room verifier.

## Validation boundary

Task and verifier validation is separate from model observation:

```sh
make harbor-check
make harbor-oracle DATASET=agent-workflow-v1
```

The suite module checks that each member ID names a direct Harbor task bundle
and validates the generated task digests. Wrong
answers, malformed or escaped evidence, incomplete scope, and false
certification receive zero reward.

## Evaluation roles

There are two supported Jacobian observation modes. Standalone observation asks
whether an agent can discover and use Jacobian. Paired control/treatment asks
whether Jacobian changes outcomes.

For a paired run, control and treatment must use identical task bundles and
task digests. The only intended difference is Jacobian availability. Do not add
Jacobian to task TOMLs: that changes the task contract and invalidates the
matched boundary.

Use the [run agent evaluations how-to](../how-to/run-agent-evaluations.md) for
commands, Docker and proxy setup, external MCP configuration, and
troubleshooting.

## Metrics and interpretation

Report mathematical correctness, evidence validity, scope/completeness, false
certification, and assurance calibration separately. Aggregate reward may
summarize a workflow contract, but is not primary evidence of Jacobian's
mathematical capability value when it combines those dimensions.

Start comparative work with three representative cases and three repetitions
per condition. Stronger claims require held-out or transformed cases, a
non-ceiling control pilot, more repetitions, and uncertainty reporting. Public
suite results remain workflow observations, not held-out causal evidence.

Inspect Harbor ATIF together with Jacobian telemetry for discovery,
descriptions, invocation and parameter errors, artifact and verification-record
flow, repeated calls, shell activity, tokens, time, and completion. This is
workflow evidence, not a causal comparison: the public suite has no held-out
performance claim.

The separate
[`research-diagnostics-v1`](../../benchmarks/datasets/research-diagnostics-v1/README.md)
dataset is public and answer-visible. Its results remain case-level diagnostics
and must not be reported as held-out model performance.
