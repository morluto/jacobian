---
name: harbor-benchmarks
description: Build, validate, and run Jacobian evaluations packaged as Harbor datasets. Use when authoring or changing Harbor tasks, independent verifiers, Oracle jobs, workflow fixtures, Jacobian observation jobs, task digests, or evaluation handoffs; keep generic Harbor CLI mechanics in the shared Harbor skills.
---

# Harbor Benchmarks

Use this skill as Jacobian's repository-specific layer for turning mathematical
cases into reproducible Harbor evaluations. It owns task contracts, verifier
integrity, dataset identity, Oracle validation, and Jacobian observation
configuration; it does not prescribe the mathematical strategy an evaluated
agent should use.

For detailed verifier design, adversarial fixture coverage, diagnostic scoring,
and evaluation-integrity review, also use the repository's
`verifier-evaluations` skill. Keep this skill focused on Harbor packaging,
dataset identity, repository gates, and observation plumbing.

## Choose the evaluation boundary

Classify the work before editing a task:

- **Task/verifier validation:** parse the bundles, run the Oracle, and attack
  deliberate failure cases. This is harness evidence.
- **Regression or public reproduction:** replay a known or public case to
  detect breakage. This is not held-out evidence and must not support a causal
  capability claim.
- **Assurance or contract conformance:** test assurance calibration, scope,
  schemas, artifacts, discovery, or parameterization. These cases may measure
  workflow quality without measuring mathematical capability value.
- **Jacobian workflow observation:** use the committed Harbor observation job,
  its direct local MCP service, and Harbor ATIF plus Jacobian telemetry. This is
  workflow evidence, not comparative performance.
- **Future causal comparison:** put control/treatment conditions in Harbor job
  configuration, outside task bundles, with identical task digests, prompts,
  models, budgets, environments, and seeds. Require held-out or transformed
  cases, a non-ceiling control pilot, multiple repetitions, and separately
  reported correctness and assurance metrics. Do not treat an A/B run on the
  public suite as a causal result.

## Author or change a task

Read `AGENTS.md`, `CONTRIBUTING.md`, and
`docs/reference/capability-workflow-evaluations.md`. Each Harbor dataset owns
its task bundles directly under
`benchmarks/datasets/<dataset>/<task-id>/`; a bundle is a direct child of its
dataset, not a symlink or nested task directory. Keep domain-owned execution
metadata in typed `task.toml`. A dataset selects its bundles through one
authoritative `benchmarks/datasets/<dataset>/members/<task-id>.toml` record per
member. Immutable snapshot locks are evaluation and publication boundaries;
never commit a mutable `dataset.toml` in a dataset root or maintain a second
shared task root.

Before adding a task, run the benchmark planner and check for a global ID
collision. Manually authored or substantially transformed cases remain authored
tasks; create `benchmarks/adapters/<source>/` only when a pinned external source
can be converted reproducibly, with source revision and digest, license status,
included/excluded rows, deterministic conversion, pinned dependencies, Oracle
evidence, and parity evidence.

Every task has a maintainer `README.md`, frozen offline input, schema 1.4
metadata, concise provenance, an agent-visible
`environment/submission_schema.json`, hidden Oracle solution material, and a
separate clean-room verifier. Instructions must
be agent-agnostic: describe the mathematical outcome and evidence, never
capability IDs, tool sequences, preferred decompositions, or Jacobian details.

Treat the agent-visible schema and instructions as the task's public protocol.
They must state the required output shape and types, allowed enum values,
assurance ceiling, evidence paths and digest rules, scope, completeness, and
artifact names. Do not rely on maintainer-only metadata or a README the agent
does not receive for constraints that affect a valid submission. Keep expected
solutions, hidden verifier logic, private authorization records, and Oracle
fixtures out of that protocol; the verifier must independently enforce every
public rule.

For a new task, copy the member shape from `benchmarks/templates/member.toml`,
create exactly one `members/<task-id>.toml` record alongside the direct task
bundle, and resolve its `environment_profile` through
`benchmarks/environment-profiles.toml`. Standard profiles use digest-pinned
images and prohibit `apt-get`; provider-specific installation belongs only in
an explicitly allowed profile. Harbor verifier Dockerfiles are built from the
task's `tests/` directory, so do not use parent-directory `COPY` paths, host
paths, floating image tags, or symlinks to share hidden material.

Verifiers must reject malformed submissions, symlink or workspace escapes,
wrong evidence paths or digests, incomplete scope, mismatched claims, and false
`VERIFIED` assertions. Score protocol compliance, correctness, evidence
validity, scope accuracy, assurance calibration, and aggregate reward; force
reward to zero for wrong answers and false certification. Accept alternate
mathematically valid
witnesses where the task permits them. Report protocol or assurance failures
alongside mathematical correctness so an aggregate zero is not misread as a
wrong mathematical answer.

Bind `/app/input.json` to the sole bounded regular frozen input before parsing
or indexing visible data. Evaluate the mathematical source from the frozen
`/tests` copy after binding; task bundles may use a task-specific frozen input
filename rather than `/tests/input.json`. Reward-bearing prose evidence must
have visible mathematical obligations and reject unrelated nonempty text.

Keep Jacobian out of task bundles. Attach it only through the Harbor job's agent
configuration and MCP sidecar. Keep credentials, raw caches, host paths,
floating dependencies, and Oracle/verifier material out of agent-visible
files.

## Validate identity and behavior

Use the pinned Harbor runner from the repository:

```sh
uvx --from harbor==0.20.0 harbor --version
make harbor-plan BASE=origin/main
make harbor-check-task DATASET=agent-workflow-v1 TASKS="task-id"
make harbor-oracle-task DATASET=agent-workflow-v1 TASKS="task-id"
```

The selected-task commands are the normal leaf-task gates and require an
explicit task selection. Use the full `make harbor-check` and explicitly
scoped `make harbor-oracle` paths for shared tooling, schemas, registry, suite
policy, or other control-plane changes. A full dataset Oracle requires
`FULL=1`; ordinary Oracle runs require `TASKS` and never expand an omitted
selection implicitly.

After any input, instruction, metadata, verifier, dependency, image, or task
contract change:

1. Recompute each prospective task and suite digest with Harbor's task model.
   Do not rewrite an existing snapshot or historical evaluation. Create a new
   snapshot only for an intentional evaluation or publication event.
2. Parse/check every canonical task selected by member fragments and reject
   missing, duplicate, ambiguous, or escaping references.
3. Run the planner-selected Oracle scope and require full applicable reward.
   For a task-local change, run every selected task. For shared support or a
   broad mechanical fan-out, run `make harbor-check`, the hand-written changed
   task Oracles selected by the planner, and leave capped full-dataset sweeps to
   the merge queue unless the operator explicitly requests a local full sweep.
   Report any deferred Oracle scope as a proof gap.
4. Exercise deliberate failures: empty or malformed output, malformed and
   wrong-shaped visible input, wrong answers,
   forged or escaped evidence, incomplete scope, mismatched claims, timeouts,
   false assurance, and correct witnesses with unsupported assurance claims.
5. Confirm alternate valid witnesses pass and scan task bundles for leakage,
   secrets, host paths, raw caches, and floating dependencies.

After generated verifier Python or validation tests change, run the repository
format check (`make lint-full` or the planner-selected equivalent), not only
`ruff check`; lint success does not imply Ruff formatting success.

### Validation regression layout (`agent-workflow-v1`)

Put new or changed task verifier attack tests in a **per-task leaf module**:

`benchmarks/validation/agent_workflow_v1/test_<task_id_with_underscores>.py`

Do **not** append to a shared `test_task_regressions_*.py` dump (those files
are gone; do not recreate them). Suite-wide contracts stay in
`test_generic_verifier_contracts.py` (assurance / fail-closed attacks over
`VERIFIER_TASKS`) and the small fixed samples in
`test_result_json_evidence_policy.py`. Edit `RESOURCE_DERIVED_TASKS` or
`VERIFICATION_RECORD_TASKS` in `support.py` only when the task shares that exact
assurance or scoring contract. Keep task-local prepare/bind helpers in the leaf;
do not grow a shared fixture module every PR edits.

Do not treat `harbor sync` as a local digest calculator when the task is not
published. Membership is authoritative in the member record and task content
is hashed by Harbor's `Task.checksum`; there is no mutable dataset-root
manifest to regenerate. `harbor-check` is outside the product `tests/`
topology; keep Harbor validation from entering product Python coverage.

Only create a snapshot for an intentional evaluation or publication boundary:

```sh
make benchmark-snapshot DATASET=agent-workflow-v1
make benchmark-snapshot-validate \
  LOCK=benchmarks/snapshots/<dataset>/<digest>.lock.json
make benchmark-publish \
  LOCK=benchmarks/snapshots/<dataset>/<digest>.lock.json
```

Commit the lock under `benchmarks/snapshots/<dataset>/`; publication output
under ignored `dist/harbor/` is generated and must not be edited or committed.
Historical plans, ledgers, observations, and reports reference the lock ID and
must never be rewritten when a later task is added.

Observation evidence is version 2. Jobs must select exactly one of
`datasets[].path` (with optional task filters) or explicit `tasks[].path`;
mixed, empty, unknown, escaping, and implicit full-suite selections fail
closed. Bind evidence to the snapshot ID, task digest, Harbor version,
normalized arguments, runtime state, model, repetition, budgets, result, and
verifier status. Use Harbor's artifact manifest as the source of truth for
artifact identity and reject traversal, escaping symlinks, missing entries, and
non-conclusion execution states.

Current separate-verifier tasks retain synchronized `tests/verifier_support.py`
copies because Harbor requires the separate verifier image to contain its test
runtime in the task `tests/` build context. Do not delete those files or the
sync checker until a digest-pinned shared verifier image is published and the
task Dockerfiles have migrated to that image. A local-only image or invented
digest is not a valid substitute.

The independent benchmark planner emits `run-benchmark-check`,
`run-benchmark-oracle`, `benchmark-oracle-scope`, an exact dataset/task/digest
matrix, and reasons. README-only task changes run contract checks without
Docker. Executable task changes run the exact task Oracle on pull requests.
Large multi-task edits are capped on the pull-request critical path and defer
their Oracle matrix to the merge queue.
Dataset membership and execution configuration changes defer their affected
dataset sweep to the merge queue; shared tooling, schemas, adapters, and unknown
integration changes escalate there to the full portfolio. Main pushes repeat
contract checks without replaying merge-queue Oracles. Scheduled, manual, and
`ci:benchmark-full` runs own explicit full-portfolio sweeps. The stable
`Benchmark Validation` workflow job is the only required branch-protection
context; dynamic Oracle jobs run at most four in parallel and upload result JSON,
verifier logs, source SHA, task digest, and Harbor version.

## Run Jacobian observation

Review the committed observation job and run the local Harbor composition only
as an explicit operator evidence exercise. The Jacobian service is intentionally anonymous for this local evaluation path;
Harbor connects directly to `http://jacobian:8000/mcp`. Set `JACOBIAN_IMAGE`
only when overriding the default local image. Use the toggle explicitly:

```sh
export JACOBIAN_MODEL='your-model'
export JACOBIAN_IMAGE='jacobian:local'
make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=1 EVAL_EXECUTE=1
```

For a control run, use `JACOBIAN_ENABLED=0`; it selects the matching Harbor
job without the sidecar or MCP configuration. Keep the task filter, model,
prompt, budget, and environment fixed when comparing the two modes. The
external MCP configuration belongs to the treatment job, not task TOMLs.

Inspect Harbor ATIF together with Jacobian telemetry for capability discovery
and descriptions, invocation and parameter errors, artifact and verification
record flow, repeated or irrelevant calls, shell/file activity, tokens, time,
cost, and completion. Record the git tree, task digests, provider/runtime,
model/settings, prompt, seeds, raw traces, and structured reports.

## Handoff and publication

Report whether the result is task validation, a public regression, workflow
observation, or a causal comparison. Include exact commands, task digests,
Oracle/verifier identities, runtime state, raw artifact locations, validation
actually run, contamination limits, and open obligations.

Keep the dataset usable directly from the repository. Publishing to a Harbor
registry is optional and requires an explicit request; registry publication is
not part of task validation or local Jacobian observation.
