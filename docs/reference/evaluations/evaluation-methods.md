# Evaluation methods

[Documentation home](../../index.md)

Jacobian evaluations measure whether the atomic mathematical toolbox improves
agent outcomes. They are explicit operator-run evidence exercises, not runtime
features and not routine pull-request gates.

## Control and treatment

Use two conditions:

```text
control:   no Jacobian
treatment: Jacobian MCP only
```

Do not bundle a Jacobian Skill, prescribed prompt, preferred call sequence, or
workflow policy into the treatment. Hold the model, reasoning effort, task
input, budget, environment, and repetition count fixed.

Harbor task inputs, hidden verifiers, and Oracle jobs remain separate from the
runtime product. Freeze task and environment digests, pin dependencies and base
images, and keep hidden solution/verifier material out of the agent-visible
environment.

## Evaluation roles

Keep evidence roles explicit:

- the evaluated model produces an answer and tool trace;
- deterministic task validation checks the public submission contract;
- the hidden verifier scores mathematical correctness and required bindings;
- Oracle validates that the task is solvable and discriminating; and
- operator review interprets aggregate results and limitations.

An evaluator score, model answer, solver status, or search result is not a
verification record. The evaluation may reward a valid bound checker record,
but cannot promote its own score to `VERIFIED`.

## Outcome metrics

Report at least:

- mathematical correctness;
- useful intermediate values;
- honest scope, completeness, and non-conclusions;
- valid/invalid Jacobian calls;
- discovery-to-execution continuation;
- irrelevant calls and fallback behavior;
- producer/checker independence where checking is attempted;
- tokens, elapsed time, and model-visible tool bytes; and
- task/verifier failures separately from model failures.

Do not reward adherence to a fixed operation order. A correct no-tool solution
may be appropriate for an easy task; negative controls and tasks with a real
mathematical affordance distinguish that case from poor discoverability.

## Reproducibility

Bind every run to:

- dataset and task digests;
- exact Jacobian source/package and advertised catalog digest;
- model and agent runtime versions;
- MCP configuration and tool schemas;
- provider identities;
- budgets and timeouts;
- random seeds where applicable; and
- raw trace/output digests.

Report missing, stale, truncated, or unavailable evidence as unknown rather
than an empty or negative observation.

### Lean diagnostic recovery ablation

`benchmarks/config/lean-diagnostic-recovery-v1.json` is a targeted engineering
ablation, not the no-Jacobian toolbox outcome comparison above. It freezes five
deliberately broken Lean proof/request injections across CORE, MATHLIB,
proof-edit validation, and proof-state source-mode validation. The premise case
records its malformed-prefix failure as a separate diagnostic probe while its
primary recovery metric is anchored to a proof rejection observable in both
conditions. It then compares the recorded base revision with a candidate revision
using `benchmarks.tooling.lean_diagnostic_recovery`. Both conditions expose only
the Jacobian MCP surface; no Skill, prescribed workflow, or retry policy is
added to either condition. The result supports descriptive diagnostic-recovery
observations only and does not authorize a causal or held-out product claim.

Model execution requires the explicit `--execute` guard. Hold model, reasoning
effort, tool mode, timeout, case selection, and repetitions fixed. Each run
prints its `report_sha256` trust anchor; retain that line outside the editable
output directory in an append-only operator log or signed manifest. Then
compare the two `report.json` files with `--compare CONTROL TREATMENT`,
`--control-report-sha256`, and `--treatment-report-sha256`. An adjacent digest
file inside either output directory is not an external trust anchor. Pass the
exact selected release revision as `--deployed-revision`. The control must
match the suite's `source_base_revision`; the treatment must match the
candidate checkout running the harness. Managed deployments expose their
root-owned release marker through `deployment://identity`; execution refuses to
start unless that endpoint-observed full Git revision agrees with both the
operator argument and the condition's source revision. This prevents a stale or
swapped endpoint from being mislabeled as the selected release.

Comparison fails closed on condition or run-plan drift, wrong source bindings,
identical observed surfaces, incomplete case/repetition coverage, or stale
summaries. Before parsing a report, it verifies the raw report bytes against the
externally retained trust anchor. It then resolves each command receipt,
transcript, and stderr file relative to that anchored report, rejects path and
symlink escapes, verifies every artifact SHA-256 digest, derives process
completion from the canonical command receipt, validates and reparses the
retained JSONL transcript, and reruns the recovery classifier for the bound
suite case. Reported call, token, rejection, diagnostic, repetition, and repair
metrics must equal the recomputed values before any delta is emitted. The
selected suite bytes and retained MCP surface are also rehashed.

The exact injected payload may appear anywhere in the freely composed tool
trace, but it must produce proof-specific rejection evidence or a retained
Lean-owned request diagnostic before a later checker-backed success can count
as repair. Failed `math.run` attempts retain bounded diagnostic codes in
evaluation telemetry; these metrics never enter runtime responses. Whether the
first injected attempt was the first Jacobian attempt is retained as a separate
descriptive protocol field and never gates
repair success. Runtime setup, toolchain, Mathlib-manifest, and timeout failures
remain non-conclusions. The terminal result must preserve each case's immutable
claim fields. Recovery metrics remain evaluation artifacts and never enter an
agent-facing runtime response.

## Performance benchmarks

Correctness gates and performance measurements remain separate. Use repeated
warm and cold runs, preserve raw samples, and report medians and tail latency
with the exact workload and environment.

Representative groups include:

- startup imports and installed catalog construction;
- canonical encoding and hashing;
- ordinary operation parse/preflight/execute/publication;
- checker plan construction and bounded replay;
- request-local and durable value transport;
- MCP stdio and HTTP round trips; and
- SQLite BLOB storage at 1 KiB, 100 KiB, 1 MiB, and 10 MiB with concurrency
  1, 4, and 16, including crash/restart and backup/restore.

For Lean proof-state backend comparisons, run
`benchmarks.tooling.lean_repl_backend_benchmark` as a pyperf cell for each
`CORE`/`MATHLIB`, `clean`/`persistent`, and prefix-length combination. The
persistent candidate preserves the same atomic reconstruction-and-one-tactic
contract and is not an agent-facing operation. Compare same-host JSON outputs,
retain corpus digests and correctness checks with latency, and pass
`--inherit-environ JACOBIAN_LEAN_BENCH_ENVIRONMENT,JACOBIAN_LEAN_BENCH_BACKEND,JACOBIAN_LEAN_BENCH_PREFIX_LENGTH`
so pyperf workers receive the selected cell identity. This is operational
evidence only and does not relax the final `lean.check` boundary.

Performance never relaxes validation, evidence binding, resource admission, or
checker independence. A fast checker that accepts forged evidence is broken.

## #905 evaluation

Slice A tasks must require the exact finite-field presentation, explicit
restriction of scalars, the `F₂⁴ → F₂⁶` map, nine direction-bound ranks, and an
orbit distribution. Trap cases use differently presented isomorphic fields,
wrong axes/bases, rank substitutions, and missing directions.

Slice B tasks require reuse of Slice A field identity and codecs, a complete
finite polynomial map table, exact fiber partition, and collision/permutation
certificates. Trap cases forge or truncate enumeration and substitute the map
or parent.

Evaluate producer and checker code paths separately. They may share passive
formats and expected digests, never executable enumeration, conversion, or
rank logic.

## Interpretation

Public-suite results are regression observations. Strong causal claims require
held-out or transformed cases, non-ceiling controls, repeated runs, uncertainty
reporting, and a frozen analysis plan. Report coverage as upstream
reproduction, meaningful authored task, or source reference rather than
combining those categories.

Any change to an executable task contract, verifier, dependency closure, or
base image invalidates prior Oracle evidence and requires rerun under the exact
task validation path.
