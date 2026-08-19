---
name: harbor-benchmarks
description: Build, validate, and run Jacobian evaluations packaged as Harbor datasets. Use when authoring or changing Harbor tasks, independent verifiers, Oracle jobs, workflow fixtures, task digests, or evaluation handoffs.
---

# Harbor Benchmarks

Build a benchmark as a small executable experiment about a difficult,
mathematically meaningful capability. Choose tasks that are indicative of the
reasoning needed for higher-level conjecture work, then make the bounded claim
replayably checkable. A task need not be solvable with Jacobian's current
operations: failures can identify the operations or environment design that
agents need next. `math.find` and `math.run` are experimental interventions,
not admission criteria. A hidden deterministic verifier decides whether the
submitted mathematical outcome satisfies the task. Do not turn Harbor into a
workflow engine, a trace grader, or a second Jacobian product surface.

For verifier design, use `verifier-evaluations` with this skill.

## Classify before authoring

- **Benchmark task:** a bounded, agent-facing mathematical claim that measures
  a meaningful high-level capability and has a replayable verifier. Put it in
  Harbor, whether it is a current tool-assisted success, a tool-free baseline,
  or an identified capability gap.
- **Integration fixture:** a pinned provider or environment feasibility spike.
  Put it under `tests/fixtures/providers/`, with boundary tests.
- **Diagnostic/public reproduction:** a known-answer regression or a
  human-reviewed proof corpus. Keep it out of comparative Harbor scoring.

Do not promote a task merely because it has a Docker image, an Oracle, or a
function call. A comparative task must test the mathematical capability it
claims to measure and must not be satisfiable by a shortcut, an empty output,
or parroting the public operation description. Specify the available Jacobian
operations as an experimental condition, including a tool-free baseline when
that comparison answers a question; do not prescribe a proof strategy or a
particular call sequence.

## Design the task

Use one of these submission shapes:

1. **Replayable result:** submit a typed `result`; the verifier recomputes or
   checks the claim from frozen input. This is the default.
2. **Constructive result:** submit `result` and a finite task-specific witness
   such as a counterexample, matrix, construction, or proof trace. The verifier
   checks it directly and accepts every declared equivalent witness.
3. **Formal proof:** use only a deliberately supported formal language and a
   real checker. Do not score natural-language proof prose automatically.

Put small structured mathematical certificates in `result`. Declare a witness
artifact only when an external finite object is genuinely needed for replay;
it must not duplicate `result` or carry a narrative explanation. Do not add
generic `claimed_assurance`, scope, completeness, limitations, or proof prose.
Ordinary mathematics tasks do not need authorization claims.

A typed result represents a mathematical value, not one JSON or textual
rendering. Normalize and compare equivalent results—such as unreduced
rationals, scaled rational functions, and unordered factors—unless the public
task explicitly makes canonicalization an outcome. Declare every intentional
normal-form or ordering rule in the public instruction and schema. Never use
`answer.txt` as the authoritative answer channel.

Do not copy these authoring failures into a new or generated task:

- scoring equality with hidden `tests/expected.json` while ignoring frozen
  `input.json`;
- requiring lowest terms or one canonical spelling while the verifier
  constructs `Fraction` or an unordered map;
- encoding exact rationals or formulas as privileged strings;
- scoring prose with keywords, length, or negation regexes;
- publishing a universal certificate `oneOf` for every generated family;
- requiring a witness that only mirrors, hashes, or paraphrases `result`;
- leaving instruction, schema, and verifier mutually unsatisfiable;
- putting a leaked answer constant or derived conclusion in the public schema
  (`const` status fields, semantic-relation enums, rationality conclusions,
  `Claim COMPUTED`, tautological bound certificates);
- requiring JSON `null` for fields a closed success variant does not declare;
  omit those fields;
- excluding Harbor host-verifier tests from unsupported-surface AST or text
  scans merely because they spawn the verifier subprocess.

The verifier replays the advertised predicate from the frozen input. Keep
`expected.json` as an Oracle fixture only. Generated families emit the
smallest schema licensed by that task's claim type. The agent-visible
instruction and schema are the complete public protocol: required result
fields, exact types, scope, and any task-specific witness rule. Do not hide a
validity requirement in a README or task metadata. Keep solution, verifier,
Oracle, host paths, and caches out of the agent environment.

When a schema shrinks, update the instruction, verifier exact-key set, gold
submission, public contract, and host tests in the same change. Align a
mismatch by shrinking the verifier and instruction, never by restoring leaked
constants. If one submitted object implies a dimension or bound, parse every
related object at that derived size. After merging `main`, re-read every
touched `instruction.md`: truncated sentences, leftover contract fragments,
and `COMPUTED` clauses on result-only schemas are protocol bugs.

Each task is a direct child of its dataset with one authoritative member record:

```text
benchmarks/datasets/<dataset>/<task-id>/
benchmarks/datasets/<dataset>/members/<task-id>.toml
```

Use frozen offline inputs, pinned images and dependencies, and an explicit
environment profile. Verifier Dockerfiles build from `tests/`; do not use
parent-directory `COPY`, host paths, floating tags, or symlinks.

## Implement the verifier

Treat the final typed mathematical result as the task outcome. The verifier
must not infer correctness from an agent's tool transcript, console output, or
claimed method; it must:

- bind the public input to the frozen verifier input before parsing it;
- validate exact types, shapes, bounds, paths, and cardinality before indexing,
  hashing, or computing;
- replay the mathematical predicate from the frozen copy;
- check only declared witnesses, scopes, and authorizations;
- write a deterministic reward artifact for malformed input as well as success.

The default task reward is binary: `1` exactly when the replayed mathematical
predicate and every required witness condition hold, otherwise `0`. Separate
diagnostic dimensions only when they describe a declared task boundary (for
example, input binding or a witness check); they do not create fractional
credit. Use non-binary scoring only for an explicitly decomposed task whose
independent subclaims are each meaningful, replayable mathematical outcomes.

For `input_binding_decoupled` tasks, keep `load_submission()` the strict
protocol loader. A bounded raw parse may support independent diagnostics, but
it never establishes public protocol validity or reward eligibility. Store
every such exception in that task's closed `tests/verifier_contract.json`;
never use a global task-name registry. If correctness or witness validity must
remain observable when `/app/input.json` is replaced, parse with
`require_input_binding=False`, replay math against the frozen tests input, and
AND binding only into `reward`. Do not fold binding into `correctness` or
`witness_validity`.

Bound raw submissions and visible/frozen inputs before parsing. If a task needs
a witness artifact, publish a finite bound only when its encoding or task
mechanics justify one; never impose a universal default merely to copy a
result. Require exact relative paths, regular non-symlink files inside the
verifier workspace, the declared digest, and a semantic connection to the
submitted claim.

## Validate

Read `AGENTS.md`, `CONTRIBUTING.md`, and
`benchmarks/docs/benchmark-contracts.md`. Use the pinned runner and planner:

```sh
uvx --from harbor==0.20.0 harbor --version
make harbor-plan BASE=origin/main
make harbor-prepare-task DATASET=<dataset> TASKS="<task-id>"
make harbor-validate-task DATASET=<dataset> TASKS="<task-id>"
```

`harbor-prepare-task` is the scoped mutating step: it formats selected task
Python and refreshes public-contract and verifier checksums. For older contracts
that the modern prepare workflow does not support, use the dataset's own check
path and scoped `tools/sync_harbor_verifier_support.py`; do not rewrite a task
only to satisfy the helper.

Static contracts, executable host validation, and Oracle replay prove different
things. After a task change, run the planner-selected static and host checks,
the exact task Oracle, and attacks for malformed output/input, wrong results,
wrong types, alternate witnesses, path escape, and every declared witness or
authorization rule. For a shared verifier migration, run the generic matrix and
the affected selected-task Oracles; report any deferred full sweep as a gap.

Treat task-local `tests/verifier_support.py` copies as authoritative because
Harbor's separate verifier image needs them in its build context. Migrate copies
explicitly, inspect their diffs, and refresh only their Dockerfile checksum
labels. Do not silently synchronize all tasks from a global runtime helper.
`verifier_bundle_checksum()` hashes filenames, NUL separators, and bytes; do
not concatenate `sha256(verifier.py)` with `sha256(verifier_support.py)`.
Refresh selected tasks with `make harbor-prepare-task` or scoped
`tools/sync_harbor_verifier_support.py`. Gold witness descriptors must resolve
to regular files under a non-symlink `solution/` root.

Create a snapshot only for an intentional evaluation or publication boundary:

```sh
make benchmark-snapshot DATASET=<dataset>
make benchmark-snapshot-validate LOCK=benchmarks/snapshots/<dataset>/<digest>.lock.json
```

Do not create or retain a snapshot merely because task contracts changed.

## Report

State whether the result is task validation, public regression, workflow
observation, or a causal comparison. Include the command, task digest, selected
scope, Oracle/verifier result, and actual proof gaps. Do not call a public
reproduction or fixture a held-out benchmark result.
