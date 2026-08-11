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
