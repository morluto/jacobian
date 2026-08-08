# Hard-task interface study PR1

This bounded study asks whether small Jacobian interface changes improve weak
Codex coordination on difficult tasks without changing the mathematics. The
frozen contract is
[`benchmarks/config/hard-task-interface-pr1.json`](../../../benchmarks/config/hard-task-interface-pr1.json).

The three tasks were already present at upstream revision
`7edc7ba9035e5f9de5a04f406cfe50e7da28d8e1`. Their Harbor task digests, model,
prompt, two-rollout count, timeout, metrics, invariants, and stop rules were
recorded before the baseline. Baseline and treatment each comprise six planned
rollouts in declared task order. A completed rollout is never retried merely
because its mathematical answer is rejected.

The host has no Docker-compatible runtime. The study therefore uses the locally
authenticated Codex CLI against an isolated public workspace and a local
Jacobian MCP server, then runs the unchanged task-owned verifier after model
exit. This is public host-local exploratory evidence, not a Harbor result and
not causal evidence.

Raw transcripts, MCP logs, isolated workspaces, verifier records, commands,
timestamps, and manifests are retained under
`benchmarks/results/hard-task-interface-pr1-{baseline,treatment}/`. The raw
result directories are intentionally not source-of-truth product contracts.

External design references are limited to current official documentation. MCP
recommends structured tool results and actionable tool-execution errors so a
model can self-correct; Anthropic recommends explicit relationships and strict
input/output descriptions in agent-facing tool contracts. These ideas may
motivate a candidate, but Jacobian trajectories and typed contracts decide what
is implemented.

The relevant references were the MCP specification's
[tool result and execution-error contract](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
and Anthropic's
[tool-design guidance](https://www.anthropic.com/engineering/writing-tools-for-agents).
Broader recommendations about splitting operations or prescribing workflows were
not used because they conflict with Jacobian's atomic, agent-owned composition
model and were not supported by these traces.

## Completed execution

The frozen execution used Codex CLI 0.147.0 with `gpt-5.4-mini`, medium reasoning,
and no web access. Baseline ran from 2026-08-08T14:36:00Z to 14:53:06Z in tmux
session `jac-interface-baseline`; treatment ran from 14:58:34Z to 15:14:34Z in
`jac-interface-treatment`. Each condition completed the planned six rollouts. The
inspection and validation sessions were `jac-interface-inspection` and
`jac-interface-tests`; their host logs are `/tmp/jac-interface-*.log`.

An initial wrapper import failed before any model process started. It was repaired
once and is not counted as a rollout. The baseline manifest's wrapper digest was
read after the wrapper had been edited for the future treatment, so it is not an
exact digest of the in-memory baseline wrapper. The saved per-run workspaces and
task/verifier digests remain available for auditing. This provenance limitation,
the lack of Docker, and the small unpaired sample prohibit a causal claim.

| Task | Baseline | Treatment | Coordination observation |
| --- | --- | --- | --- |
| `graph-artifact-composition` | 2/2 accepted; 9 tool errors; 0 successful math invocations; 1/2 complete reasoning lifecycles | 2/2 accepted; 0 tool errors; 2 successful invocations; 2/2 complete lifecycles | The terminal score was already saturated, but the treatment used one exact distance-matrix call per run with no repeated calls or protocol errors. |
| `polynomial-normalization` | 0/2 accepted; rewards 0.0 and 0.9 | 0/2 accepted; rewards 0.0 and 0.9 | Setup-invalid in both conditions; descriptive evidence only. One treatment followed the advertised producer-to-checker relation with one fewer discovery search, but then persisted the checker summary instead of the verification-record payload and falsely claimed `VERIFIED`. |
| `symbolic-coordination-semantic-equivalence-01` | 1/2 accepted; 9 tool errors; 2 successful checker invocations | 0/2 accepted; 8 tool errors; 0 successful invocations; 1 repeated call | Both treatment runs recovered to a complete log but misread cancellable/reordered sparse terms and incorrectly rejected the valid inverse. |

The nominal aggregate changed from 3/6 accepted to 2/6, but the polynomial rows
are invalid observations. Across the four valid graph and semantic-equivalence
rows, acceptance changed from 3/4 to 2/4. Scope and completeness declarations did
not create a false certification in those valid rows. No rollout was retried, and
no timeout or incomplete search was interpreted as a mathematical conclusion.

## Failure taxonomy

- The baseline repeatedly used phase-incompatible reasoning fields, fabricated
  call identifiers, or finalized before completing a capability cycle. Explicit
  phase field sets removed all such errors in the two graph treatment runs, but a
  semantic treatment still combined `NOT_APPLICABLE` with `reported_*` fields.
- Producer/consumer artifact compatibility was real but absent from descriptors.
  The implemented graph and polynomial descriptors now publish symmetric typed
  artifact schemas and factual related-capability links. The graph task did not
  exercise that handoff; the polynomial behavior is not scoreable because its
  public contract is incomplete.
- The semantic-equivalence failures were mathematical representation failures,
  not stopping or call-frequency failures. The model failed to normalize duplicate,
  reordered, zero, and cancelling sparse terms before composition. More interface
  prose did not repair this gap.
- `polynomial-normalization` contains
  `environment/verification_record_schema.json`, and its README says the schema is
  agent-visible, but its environment Dockerfile copies only `input.json` and
  `submission_schema.json`. All four polynomial rollouts therefore lacked the
  public contract needed to persist a valid `VERIFIED` record. This benchmark gap
  is recorded but intentionally not changed in this interface PR.

## Intervention decisions

Keep the explicit `reasoning.write` phase fields: they make an existing protocol
inspectable and produced a strong graph coordination signal without changing
mathematical policy. Keep the typed artifact metadata and factual relationships as
contract corrections covered by composition and MCP tests, but treat their
behavioral benefit as inconclusive. Do not add a second PR or another model study:
the remaining semantic-equivalence failure points to a separately scoped sparse-map
normalization affordance or better validation diagnostics, while the benchmark's
missing public verification schema must be repaired independently before it can
evaluate checker handoff.
