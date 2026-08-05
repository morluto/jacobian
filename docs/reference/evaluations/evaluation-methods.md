# Evaluation methods

[Documentation home](../../index.md) · [Benchmark contracts](benchmark-contracts.md)

## Workflow observations

Jacobian's fixed workflow observation surface is the Harbor
[`agent-workflow-v1`](../../../benchmarks/datasets/agent-workflow-v1/README.md)
dataset. Its self-contained mathematical tasks cover graph, algebra,
linear-algebra, number-theory, geometry, combinatorics, probability, and
formal-mathematics workflows, including the original graph, partition, SAT,
linear-system, Hermite, and polynomial cases.

The task bundles are agent-agnostic. Instructions describe the mathematical
outcome and evidence without naming capability IDs or prescribing
decomposition, verification order, or stopping criteria. Each task freezes its
offline input, Oracle-only solution, and separate clean-room verifier.

### Evaluation roles

There are two supported Jacobian observation modes. Standalone observation asks
whether an agent can discover and use Jacobian. Paired control/treatment asks
whether Jacobian changes outcomes.

For a paired run, control and treatment must use identical task bundles and
task digests. The only intended difference is Jacobian availability. Do not add
Jacobian to task TOMLs: that changes the task contract and invalidates the
matched boundary.

Use the [run agent evaluations how-to](../../how-to/run-agent-evaluations.md)
for commands, Docker and proxy setup, external MCP configuration, and
troubleshooting.

### Running workflow observations

Use the fixed `agent-workflow-v1` tasks for explicit operator-run Jacobian
workflow observations:

```sh
make agent-eval DATASET=agent-workflow-v1 TASKS=graph-counterexample EVAL_EXECUTE=1
```

Model execution with `make agent-eval ... EVAL_EXECUTE=1` is an explicit
operator-run evidence exercise; it is not a routine development or pull-request
gate.

The committed three-attempt control/treatment job files are reproducibility
fixtures and remain unchanged. Running model jobs is not a routine task
authoring or pull-request step; operators may run them when collecting
evidence, then validate and compare the resulting records.

Normalize each condition with `make agent-eval-validate`, passing a
`RUNTIME_SNAPSHOT` that binds the immutable benchmark snapshot ID and pinned
Harbor version, then compare the two evidence files with `make agent-eval-compare`.
The comparator rejects unmatched task repetitions or drift in task digests,
prompts, models, budgets, and job configuration. It reports correctness and
assurance separately and marks small samples as descriptive. Public suite
comparisons remain workflow observations.

### Held-out runs

Held-out C1/C2 runs use `.github/workflows/heldout-benchmarks.yml`. A protected
GitHub environment assumes a read-only S3 role through OIDC, validates the
private manifest and archive before extraction, renders both conditions from
one frozen specification, and uploads only non-Oracle results. The pilot fixes
three tasks and three repetitions; a decision run requires at least five tasks
and five repetitions. Neither report automatically authorizes a causal claim.

Use `research-diagnostics-v1` only for answer-visible diagnostic runs. Its
public source answers and Oracle summaries remain hidden from the agent
container at runtime, but their public availability permanently disqualifies
the dataset from held-out model claims.

## Metrics and interpretation

Report mathematical correctness, evidence validity, scope/completeness, false
certification, and assurance calibration separately. Aggregate reward may
summarize a workflow contract, but is not primary evidence of Jacobian's
mathematical capability value when it combines those dimensions.

The committed control/treatment jobs preserve three attempts per condition as
manual reproducibility fixtures. Starting comparative work with three
representative cases and three repetitions per condition is an operator choice,
not a required development step. Stronger claims require held-out or
transformed cases, a non-ceiling control pilot, more repetitions, and
uncertainty reporting. Public suite results remain workflow observations, not
held-out causal evidence.

Inspect Harbor ATIF together with Jacobian telemetry for discovery,
descriptions, invocation and parameter errors, artifact and verification-record
flow, repeated calls, shell activity, tokens, time, and completion. This is
workflow evidence, not a causal comparison: the public suite has no held-out
performance claim.

## Performance benchmarks

### Purpose

Performance benchmarks answer operational questions: how quickly and at what
resource cost can Jacobian store, dispatch, replay, and search? They do not
establish mathematical correctness. Every benchmark target must already pass
its contract and conformance tests.

The initial goal is a reproducible baseline, not an invented service-level
objective. Hard regression thresholds should be set only after repeated
measurements on a controlled runner show the natural variance of each
benchmark.

### Measurement method

Use `pyperf` for Python microbenchmarks and small component benchmarks. It
provides calibrated worker processes, warmups, run metadata, JSON results, and
same-host comparisons. Larger end-to-end and service benchmarks may use a
separate harness, but must follow the same rules:

- record commit, Python version, dependency lock digest, CPU, memory, operating
  system, storage type, and benchmark corpus digest;
- measure outside coverage, profilers, and debug logging;
- use warmups and multiple worker processes where applicable;
- retain raw results, not only a summary table;
- compare like with like on the same class of runner;
- report instability rather than hiding it with repeated cherry-picked runs;
- keep correctness assertions enabled around setup and final outputs.

Wall time alone is insufficient. Record, where relevant:

- CPU time;
- peak resident memory;
- bytes read and written;
- artifact and metadata storage amplification;
- operations or candidates per second;
- cold and warm cache behavior;
- p50 and p95 latency for repeated service operations;
- startup and clean-process replay cost.

### Corpus design

All benchmark inputs are immutable, versioned artifacts. Avoid one synthetic
payload shape that rewards a particular implementation.

The initial corpus contains:

- canonical exact values with small and very large numerators and denominators;
- shallow and deeply nested valid objects up to configured limits;
- artifact blobs near 1 KiB, 100 KiB, and 10 MiB;
- manifest graphs with zero, tens, and thousands of parent references within
  allowed limits;
- batches of 1, 32, and the configured maximum of 256 candidates;
- direct witnesses and finite-enumeration certificates of several sizes;
- cold-store, deduplication-hit, and verified-cache-hit cases.

These are workload points, not public size limits. Normative parser and storage
limits are chosen in the schema and storage specifications and tested as
correctness properties.

The two reference-plugin corpora include:

- tiny hand-auditable cases;
- small exhaustive cases suitable for pull-request runs;
- medium cases that exercise nightly throughput and memory;
- false candidates with short witnesses;
- valid candidates with complete finite certificates;
- shrink traces with successful, rejected, duplicate, and cyclic proposals.

The directed-graph/path corpus is one adversarial workload, not the definition
of the runtime. The second non-graph plugin supplies a different candidate shape,
witness representation, and cost profile.

The exact public workloads are defined in the
[Mathematical scenario catalog](../scenarios/math-scenarios.md).

### Runtime and capability benchmark groups

#### Canonical encoding and hashing

Measure:

- validate and canonicalize exact values;
- canonicalize complete artifact payloads;
- hash canonical bytes;
- decode and verify stored bytes;
- rejection cost for over-limit and malformed objects.

Report throughput by canonical byte and by object. Keep validation and hashing
as separate sub-benchmarks so an optimization cannot silently remove a required
check.

#### Artifact store

Measure:

- cold artifact persistence;
- duplicate artifact persistence;
- manifest commit and lookup;
- verified blob read;
- store reopen;
- concurrent idempotent insertion once concurrency is supported;
- garbage-collection mark traversal without deletion;
- quota-check overhead.

Report blob and SQLite write amplification as well as latency. Crash recovery is
a conformance test, not a speed benchmark.

#### Checker registry and dispatch

Measure:

- compatible checker resolution;
- rejected incompatible lookup;
- authorization-policy lookup;
- cold checker process startup;
- warm dispatch;
- verification-cache lookup.

The benchmark must execute the same binding checks as production. A
short-circuit benchmark-only path is forbidden.

#### Witness and certificate replay

Measure:

- direct witness validation and replay;
- finite-enumeration certificate parsing;
- per-row and total replay cost;
- clean-process verification;
- invalid-evidence rejection at early and late mutation positions.

Invalid evidence is included because adversarial inputs should be bounded.
Rejection benchmarks do not replace parser limit tests.

#### Evaluation and witness capabilities

Measure:

- fixed-overhead batch dispatch for 1, 32, and 256 trivial candidates;
- result-envelope construction and artifact persistence;
- mixed batches with accepted, rejected, timed-out, and errored items;
- cold and warm evaluation-cache behavior;
- proposed witness persistence by URI.

Report capability-adapter and runtime overhead separately from backend
evaluation time. This shows when optimizing Jacobian matters and when domain
computation dominates.

#### Shrinking

Measure:

- proposal bookkeeping per reducer result;
- preservation-checker calls per accepted reduction;
- duplicate/cycle detection;
- trace persistence;
- time to reach a verified local reduction on fixed small cases;

Report the number of checker invocations and candidates considered alongside
time. A faster result produced by skipping verification is invalid.

#### CLI and MCP capability surface

Measure:

- installed CLI startup;
- local MCP stdio startup;
- `math.find` for one installed descriptor;
- one small `math.run` request;
- batch request encoding and decoding;
- resource-handle response construction.

Adapter benchmarks compare their overhead with direct Python API calls. They
must not embed large artifacts simply to improve apparent resource-read
latency.

### Correctness benchmarks

The reference episodes in [benchmark contracts](benchmark-contracts.md) are
pass/fail research benchmarks, not performance contests. Track:

- whether the hidden semantic object is found;
- whether its witness independently verifies;
- whether corrupted and rebound evidence is rejected;
- whether the verified example replays in a clean process;
- whether shrinking reports an honest minimality level without trusting an
  empty reducer response;
- whether both domains use the same runtime contracts.

Runtime and resource use may be reported secondarily. Correctness is the gate.

Performance measurements remain an engineering instrument rather than a
committed Harbor evaluation dataset. When a benchmark is worth preserving,
publish it as a new versioned dataset with a pinned source revision, toolchain,
environment, and fresh Oracle evidence; never revive a historical fixture by
silently changing its runtime or task imports. Compare base and candidate runs
only on the same controlled runner with matching task, environment, dependency,
and corpus digests. Thresholds remain report-only until repeated controlled
baselines establish natural variance and the project explicitly promotes a
metric to a gate.

### Bounded-discovery benchmarks

Measure:

- raw and isomorphism-reduced candidates per second;
- canonicalization cache hit rate and cost;
- durable progress-snapshot and cancellation overhead;
- exact separator and projection cost by dimension and generator count;
- transformation proposal and verification cost separately.

Enumeration benchmarks always report the exact declared scope and number of
unique canonical objects. Throughput without scope correctness is meaningless.

The executable alpha harness measures canonicalization and exact finite
polytope proposal costs.

### Portfolio and growth benchmarks

These groups are independent measurement areas, not sequential release
milestones. Add a benchmark when an implemented capability or infrastructure
change creates a concrete performance question.

#### Search capabilities

Measure:

- candidates evaluated per backend-second;
- capability invocation and artifact-persistence overhead;
- bounded search progress and cancellation cost;
- candidate, witness, and certificate storage amplification;
- duplicate-elimination and canonicalization cost;
- cold and warm backend startup;
- batch scaling where repetitive transcript evidence justifies batching.

Establish a correct single-invocation reference run before interpreting batch
or backend-parallel speedups.

#### Claim-transformation capabilities

Measure:

- falsification throughput for generated hypotheses;
- duplicate and near-duplicate filtering cost within supplied records;
- parameter-region proposal and certificate replay cost;
- verified yield per compute budget.

No count of generated conjectures is meaningful without falsification and
trust labels. Corpus-wide novelty is not a performance claim.

#### Research corpus integration

Measure:

- ingestion throughput and storage per episode;
- exact filter, structural, formula, and text retrieval latency;
- index refresh and retention-policy cost;
- deduplication and curated-promotion overhead;
- held-out improvement to search or repair, not just query speed;
- provider-unavailable fallback overhead.

Quality metrics must be computed separately by trust label, provider corpus,
and temporal cutoff.

#### Reproducibility

Measure:

- bundle export size and time;
- offline integrity verification;
- clean-install checker resolution;
- full independent replay time.

### Agent portfolio evaluations

Tool overhead is only one part of system performance. Run paired agent
evaluations with the same model, budget, seeds, and tasks:

- baseline without the capability family and treatment with the full portfolio;
- controlled ablations for discovery, retrieval, construction, search,
  transformation, and verification capabilities;
- held-out tasks and hidden independent oracles;
- repeated trials that expose variance and tool-selection failures.

Record task correctness and false certification alongside wall time, tokens,
tool calls, parameter errors, backend calls, and artifact volume. Report
prescribed-tool contract tests separately from autonomous portfolio tests. A
faster agent that reaches a wrong or falsely certified conclusion is worse, not
better.

Use transcripts to identify repetitive calls, oversized results, confusing
parameters, missing summaries, and poor capability boundaries. These
measurements may justify examples, batching, discovery ranking, splitting, or
consolidation. They do not justify hiding intermediate mathematical artifacts
or weakening independent replay.

## Regression policy

During initial development, performance jobs publish trends but do not fail
pull requests. A benchmark becomes a gate only when:

1. its correctness behavior is already covered elsewhere;
2. its corpus and harness have stable immutable identities;
3. at least ten controlled baseline runs characterize variance;
4. the project has documented why regression in that metric matters;
5. the threshold is larger than ordinary run-to-run noise.

Potential regressions are confirmed on a controlled runner against the exact
base and candidate commits. A single noisy shared-CI result is not sufficient.
Absolute memory, output, and resource limits are different: those are
correctness and availability gates from the moment the limit is specified.

Raw benchmark JSON and corpus digests should be retained as build artifacts.
Only curated summaries belong in long-lived documentation.
