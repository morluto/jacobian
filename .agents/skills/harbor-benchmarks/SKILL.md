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
`docs/reference/evaluations/benchmark-contracts.md`. Each Harbor dataset owns
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

Before choosing the mutating workflow, confirm that the dataset uses the
current public-contract shape. Some older or provider-focused bundles have a
Harbor `task.toml` but a task-local `tests/public_contract.json` that is not
the repository's modern `PublicContract` model. Do not rewrite such a contract
just to make `harbor-prepare-task` accept it. Use `make harbor-check-task` and
the dataset's own validation path, and after verifier edits run the explicit
`tools/sync_harbor_verifier_support.py` checksum update for the affected task.
Report that the modern prepare/validate path was not applicable instead of
calling an incompatible command a validation failure in the task itself.

## Validate identity and behavior

Use the pinned Harbor runner from the repository:

```sh
uvx --from harbor==0.20.0 harbor --version
make harbor-plan BASE=origin/main
make harbor-prepare-task DATASET=mathematical-benchmarks-v1 TASKS="task-id"
make harbor-validate-task DATASET=mathematical-benchmarks-v1 TASKS="task-id"
```

For benchmark paths, treat `make harbor-plan BASE=...` as the authoritative
planner. The general `make test-plan BASE=...` may classify task bundles as
documentation because they are evaluation assets; that classification does not
replace the benchmark contract, host, and Oracle plan.

### Distinguish static contracts from executable host validation

Static benchmark validation proves repository topology, schemas, metadata,
digests, checksum labels, and other source-readable contracts. It does not
prove that a task-local verifier imports successfully or behaves correctly in
its executable host environment. Treat host validation as a separate semantic
gate, not as a repetition of static validation.

When aggregate host validation fails, collect the complete failure set before
editing shared support. Reproduce each failure through its owning per-task leaf
or the narrowest selected host-validation entry, and confirm whether the
failure follows the task-local verifier, its frozen support, or shared host
tooling. A broad command that exposes a task-local failure is not evidence that
the aggregate runner, cache, or static contract is the root cause.

Benchmark host shards must partition one deterministic collection. Every shard
must receive the same `--randomly-seed`, sourced from
`pytest_randomly_shard_seed` in `.github/ci-config.json`, before
`pytest-split` selects its group. When shard command construction changes, add
or run a focused regression proving that every generated shard command carries
the configured seed. Different per-runner collection orders can create
overlapping groups even when every individual shard succeeds.

Published duration maps are merged by test node ID. Duplicate observations are
reconciled conservatively by retaining the maximum duration and emitting a
warning so one overlap cannot discard otherwise valid timing evidence. This is
a defensive publication rule, not a substitute for deterministic shard
ownership: fix missing seed propagation or other producer-side overlap first.

`harbor-prepare-task` is the explicitly mutating authoring step. It formats only
the selected task Python and dedicated validation leaf, performs scoped public
contract and verifier checksum synchronization, and reports exactly which
generated files changed. `harbor-validate-task` is the complete source-read-only
leaf gate: it resolves membership and planner-owned host selectors once, runs
static quality before contracts, executes the selected leaf and generic tests
with an isolated worktree-local pytest directory, then runs exact task Oracles
serially and reports timings, digests, and evidence paths.

Both commands require an explicit task selection. The lower-level
`harbor-sync`, `harbor-check-task`, and `harbor-oracle-task` targets remain
available for a narrow edit loop. Use the full `make harbor-check` and explicitly
scoped `make harbor-oracle` paths for shared tooling, schemas, registry, suite
policy, or other control-plane changes. A full dataset Oracle requires `FULL=1`;
ordinary Oracle runs require `TASKS` and never expand an omitted selection
implicitly.

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
   For any resumable or augmented Oracle run, derive a deterministic job
   identity from the selected task digests and normalized execution
   configuration. Record that identity in the run manifest and require the
   validated result to live under the exact expected job identity. A changed
   digest, compose context, or relevant configuration must produce a different
   identity and must not resume or validate stale results from the previous run.
4. Exercise deliberate failures: empty or malformed output, malformed and
   wrong-shaped visible input, wrong answers,
   forged or escaped evidence, incomplete scope, mismatched claims, timeouts,
   false assurance, and correct witnesses with unsupported assurance claims.
5. Confirm alternate valid witnesses pass and scan task bundles for leakage,
   secrets, host paths, raw caches, and floating dependencies.

For a cross-cutting reward or diagnostic migration, run the complete generic
verifier matrix in addition to selected task leaves and Oracles. Leaf tests can
miss a task-local exception or a metadata-driven contract regression; the
generic matrix should cover replaced and malformed visible input, malformed
and wrong-shaped submissions, unhashable assurance values, false assurance,
evidence escapes, and aggregate-reward hard gates across all registered tasks.

Tasks marked `input_binding_decoupled` require a deliberate verifier split:

- Parse a bounded raw submission without making input binding a prerequisite so
  mathematical correctness and other independent diagnostics remain observable.
- Validate the public envelope separately with
  `load_submission(require_input_binding=False)` and
  `strict_submission_contract`.
- Read the mathematical source only from the frozen `/tests` copy, report
  `input_binding` independently, and keep input binding a hard aggregate-reward
  gate.
- Use `scope_independent_assurance` only when scope can be judged safely from
  the raw object even if `claimed_assurance` has the wrong type. Otherwise
  protocol validity remains part of the scope diagnostic.

Add generic adversarial tests whenever these metadata flags are introduced or
changed. A malformed visible input must not be reported as wrong mathematics
for a decoupled task, and an invalid assurance value must not crash the
verifier or be used in an unsafe membership or hash operation.

After generated verifier Python or validation tests change, run the repository
format check (`make lint-full` or the planner-selected equivalent), not only
`ruff check`; lint success does not imply Ruff formatting success. Ruff
formatting applies to the entire `benchmarks/` tree, including task verifier
code under `benchmarks/datasets/<dataset>/<task-id>/tests/verifier.py`.

After any verifier Python (`tests/verifier.py`) or Dockerfile change, run
`make harbor-prepare-task DATASET=... TASKS="..."` to update the verifier
checksum label embedded in each task's `tests/Dockerfile`. The
`harbor-contracts` gate rejects stale checksum labels; the preparation command
uses the scoped `harbor-sync` operations to recompute them from current verifier
source.

### Validation regression layout

Put new or changed task verifier attack tests in a **per-dataset, per-task leaf
module**:

- `benchmarks/validation/mathematical_benchmarks_v1/test_<task_id_with_underscores>.py`
- `benchmarks/validation/public_reproductions_v1/test_<task_id_with_underscores>.py`
- `benchmarks/validation/conjecture_probes_v1/test_<task_id_with_underscores>.py`

Use the validation package matching the task's owning dataset; do not route a
relocated task through its former dataset package.

Do **not** append to a shared `test_task_regressions_*.py` dump (those files
are gone; do not recreate them). Suite-wide contracts stay in
`test_generic_verifier_contracts.py` (assurance / fail-closed attacks over
`VERIFIER_TASKS`) and the small fixed samples in
`test_result_json_evidence_policy.py`. Edit `RESOURCE_DERIVED_TASKS` or
`VERIFICATION_RECORD_TASKS` in `support.py` only when the task shares that exact
assurance or scoring contract. Keep task-local prepare/bind helpers in the leaf;
do not grow a shared fixture module every PR edits.

Keep generic verifier tests generic: task-specific assurance, scope, or input
binding exceptions belong in contract metadata under that task's `tests/`
directory. Do not encode those exceptions in global task-name flag registries
inside shared support or generic tests. The generic matrix should load the
task-local metadata, exercise the same common behavior, and fail when metadata
references a missing or non-member task. When resolving a merge conflict,
reconcile the metadata against the canonical task directories so removed or
renamed tasks cannot remain as stale global entries.

Do not treat `harbor sync` as a local digest calculator when the task is not
published. Membership is authoritative in the member record and task content
is hashed by Harbor's `Task.checksum`; there is no mutable dataset-root
manifest to regenerate. `harbor-check` is outside the product `tests/`
topology; keep Harbor validation from entering product Python coverage.

Only create a snapshot for an intentional evaluation or publication boundary:

```sh
make benchmark-snapshot DATASET=mathematical-benchmarks-v1
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

Current separate-verifier tasks retain task-local `tests/verifier_support.py`
copies because Harbor requires the separate verifier image to contain its test
runtime in the task `tests/` build context. The local copy is authoritative and
is already covered by Harbor's whole-task digest; it is not compared with a
global runtime catalog or copied into tasks at validation time. The task
template supplies the generic helper only when a new task is scaffolded.
Shared support changes therefore require an explicit, selected-task migration:
update the affected local copies deliberately, run each affected Oracle, and
refresh only their checksum labels with the scoped `harbor-sync` command.

Do not add arbitrary byte limits to benchmark evidence. Evidence must still
remain schema-valid, digest-bound, at the declared path, and inside the
verifier workspace, and malformed or unbound evidence must fail closed.

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
make agent-eval DATASET=mathematical-benchmarks-v1 \
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
