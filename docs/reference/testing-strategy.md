# Testing strategy

[Documentation home](../index.md)

## Change matrix

| Change | First local check | Escalate when |
| --- | --- | --- |
| Documentation or benchmark README/validation | `make docs-linkcheck` | No Oracle is needed |
| Python behavior | `make test-plan BASE=<revision>` and the selected semantic lane | Finish with `make check` |
| Harbor job JSON, MCP config, job-level Compose overlay, or execution helper | `make harbor-execution-check` | Escalate to `make harbor-check` only when shared benchmark validation also changes; task `environment/docker-compose.yaml` changes are benchmark task input, not job overlays |
| Benchmark task input or verifier (including task `environment/docker-compose.yaml`) | `make harbor-check-task DATASET=... TASKS=...` | Run `make harbor-oracle-task ...` for the selected task |
| Deployment entrypoint | `make deploy-check` | Include affected process checks for code changes |

The four primary profiles cover most product changes. `unit` owns pure
contracts and models; `component` uses one real service or adapter; `domain`
loads explicitly selected mathematical bundles; and `composition` exercises
complete runtime wiring. Boundary profiles (`storage`, `process`, `mcp`,
`provider`, `lean`, and `e2e`) own changes whose evidence crosses those named
interfaces. Use the narrowest profile that proves the behavior, then follow the
matrix when the change also affects shared infrastructure.

## Ownership model

One reviewed manifest
([`tests/plan_manifest.toml`](../../tests/plan_manifest.toml)) is the
authoritative source for pytest lanes, gates, and path-impact rules.
`make compile-test-plan` projects it to
[`tests/topology.toml`](../../tests/topology.toml) and
[`.github/ci-impact.json`](../../.github/ci-impact.json). Rule-local
`suppresses` fields drive classifier specificity; do not hand-edit the
generated projections.

| Dimension | Answers | Authority |
| --- | --- | --- |
| Semantic owner | What assertion layer? | Test directories (`unit` / `component` / `domain` / `composition` / `e2e` + boundary seams) |
| Resources | What isolation hardware? | Typed fixture contracts (`sqlite`, `process-group`, `mcp`, `complete-runtime`, …) |
| Runtime profile | Minimum install/authority/mutability | `RuntimeTestProfile` in `tests/support/runtime_profiles.py` |
| CI policy | When does it run? | Manifest `ci` / `runs_on` + impact rules |
| Execution profile | Workers/timeout/scheduler | Compiled from the dimensions above |

Lane identity also appears in Make targets and workflow jobs. Edit
`tests/plan_manifest.toml` and regenerate rather than hand-editing
`tests/topology.toml` or `.github/ci-impact.json`.

### Hydration ladder

Use the narrowest complete-runtime profile that proves the claim:

1. `open_domain_services(bundle)` — one named domain bundle (producers only)
2. `open_exact_domain_services(bundle)` — one named domain bundle **with** its
   exact verification adapters (typed verified-domain seam)
3. `attached_complete_runtime` — complete portfolio, **no** checker authority
   (reference schemas/plugins are available without authorization)
4. `authorized_complete_runtime` — complete portfolio **with** authorized checkers
5. `fresh_complete_runtime` — empty-root install / lifecycle ownership only

`authorized_complete_runtime` requires a real verify/authority assertion in
the module (for example `CapabilityAssuranceLevel.VERIFIED`,
`services.verification`, `capability_id="….verify"`, or
`checker_id is not None`). Catalog ID strings and `UNVERIFIED` alone do not
justify it. Inventory unjustified uses with `make test-runtime-inventory`; the
inventory fails closed when any remain.

A test's directory answers what kind of behavior it owns. A marker is retained
only when it changes execution. The CI impact manifest maps changed paths to
every explicitly owned lane, with additive multi-owner rules and a fail-closed
fallback. Product Python lanes are selected independently; benchmark paths are
excluded from this control plane and use the separate Harbor planner. Make
targets keep local and hosted execution aligned. Timing data affects domain and
composition sharding only: successful `main` runs publish it, consumers validate
it, and any absence or corruption falls back to equal weighting without changing
which tests run.

The canonical local commands are the semantic targets `make test-unit`,
`make test-component`, `make test-domain`, `make test-composition`,
`make test-storage`, `make test-process`, `make test-mcp`, `make test-provider`,
`make test-lean`, and `make test-e2e`. `make test-all-ci` is an explicit,
exceptional full local reproduction.

Planning has three intentionally different entry points:

```sh
make test-plan BASE=origin/main   # exact local selectors or lane fallback
make test-plan PATHS='src/jacobian/domains/graph.py'  # hypothetical local change
make ci-plan BASE=origin/main     # hosted CI semantic lanes
make ci-plan PATHS='deploy/install.sh'                # hypothetical hosted plan
make harbor-plan BASE=origin/main # benchmark contracts and Oracle scope
make harbor-plan PATHS='benchmarks/README.md'         # hypothetical benchmark plan
make deploy-check                # deployment entrypoint syntax and boundary
```

The hosted CI and Harbor plans are evidence plans, not test commands. They
report why a lane is selected, deferred, or escalated, and emit a temporary
receipt bound to the event, base/head revisions, changed-path digest, planner
digest, configuration digests, and canonical plan digest. `make harbor-plan`
uses the pinned Harbor runtime because task digests are part of the plan
contract.

### Local development and CI ownership

The contributor quick path is `make setup PROFILE=core` followed by
`make check-changed BASE=origin/main` (see
[CONTRIBUTING.md](../../CONTRIBUTING.md)). It keeps the local loop on the
changed-path gate. CI owns the exhaustive correctness surface that the local
loop intentionally skips: the supported Python and OS matrices, the full Lean
and optional-provider environments, coverage enforcement, the compatibility
smoke suite, packaging, the security audit, duplicate-code detection, and the
complete semantic-lane matrix. You do not need to reproduce those locally for a
routine change.

Specialist lanes (`storage`, `process`, `mcp`, `provider`, `lean`, and `e2e`)
own their named boundaries, but for routine contributor work they are
troubleshooting and boundary-crossing work rather than a required local gate.
Run one when a change crosses that boundary or when reproducing an
environment-specific failure; CI runs the full matrix. The command inventory
below is the authoritative reference for lane commands, narrowing, planning
entry points, and CI classification, and is linked from
[CONTRIBUTING.md](../../CONTRIBUTING.md) instead of being duplicated there.

## Purpose

Jacobian is a mathematical capability toolbox with a fail-closed verification
boundary. Its tests must establish more than ordinary application correctness:
they must demonstrate that malformed inputs, incomplete computations, stale
caches, and substituted evidence cannot be promoted into verified mathematical
conclusions.

This document defines the test architecture for the current implementation.
Current contracts state what the implementation must do; this document states
how we build and maintain evidence that it does so.

Two principles govern the suite:

1. Tests exercise public behavior and stable artifacts, not private helper
   names or copied implementation branches.
2. Correctness gates and performance measurements remain separate. A fast
   checker that accepts invalid evidence is simply a broken checker.

`EXACT_RATIONAL + EXHAUSTIVE` is one strong evidence profile, but it does not
automatically create a verified result. A direct exact witness need not be
exhaustive, and a checked SAT proof or proof-assistant term may certify a claim
through another assurance route. In every case, only an operator-authorized
checker may originate `verification = VERIFIED`.

The current local development entry points are:

```sh
make test-unit
make test-component TESTS=tests/component/capabilities/test_atomic_capabilities.py
make test-component TESTS=tests/component/capabilities/test_atomic_capabilities.py PYTEST_ARGS="-k schema -n 0"
make test-domain TESTS=tests/domain/graph/test_graph_invariant_domain.py
make test-composition
make test-mcp PYTEST_ARGS="-k authentication"
make test-storage TESTS=tests/boundary/storage/transactions/test_state_database_migrations.py
make test-process TESTS=tests/boundary/process/search/test_shrinking.py
make test-provider
make test-lean TESTS=tests/boundary/providers/lean/test_lean_repl_runtime.py PYTEST_ARGS="-k induction"
make test-e2e
make test-stress
make test-ordering ORDERING_LANE=domain PYTEST_ARGS=--randomly-seed=17
make check
make check-changed BASE=origin/main
make check-static
make test-all-ci
make docs-linkcheck
make ci-plan BASE=origin/main
make test-plan BASE=origin/main
make harbor-plan BASE=origin/main
make harbor-execution-check
make clean
```

The Makefile pytest targets emit the ten slowest tests by default so local
commands provide actionable duration telemetry. Override this with
`PYTEST_DIAGNOSTIC_ARGS=--durations=0`, or increase the count when profiling a
resource-heavy lane.

Use the narrowest semantic lane that proves the claim. Unit tests are pure;
component tests use one real service or adapter; domain tests install only
named `DomainBundle` values; composition tests cover complete runtime wiring;
and boundary/e2e tests own persistence, processes, providers, MCP, Lean, and
complete user workflows. Directory ownership replaces the old catch-all
integration category.
Complete-runtime fixtures belong only to assertions about global catalog,
cross-bundle, checker-authority, or lifecycle wiring. A behavior that consumes
one named bundle stays in its domain lane and opens that bundle directly;
fixture scope must not be broadened to trade away mutable-state isolation.
`make test-stress` repeats only tests marked `property`, while
`make test-ordering ORDERING_LANE=<lane>` reproduces the scheduled ordering
seed for one semantic lane. `ORDERING_LANE` is required; use `domain` or
`composition` for the sharded lanes whose scheduled seed matters most.
Domain and composition timing shards use one fixed `pytest-randomly` seed from
`.github/ci-config.json`. Every shard must collect tests in the same order
before `pytest-split` partitions them; otherwise independently randomized
collections can overlap or omit tests. The merged timing artifact rejects
duplicate node IDs instead of concealing such an overlap.
`make check` combines Ruff, strict typing, and the unit lane for a local handoff.
The installed pre-push hook intentionally runs only `make lint typecheck`; the
affected-test planner and CI own resource-heavy correctness lanes. Use
`PYTEST_ARGS="-n 0"` for debugger-friendly execution and
`PYTEST_ARGS="--durations=25"` when investigating regressions. Lane timeouts
are containment policy, not performance assertions; process and native-backend
work runs in killable children where a signal-only timeout cannot interrupt it.
Immutable fixture templates are published by constructing in a temporary
sibling and atomically renaming the completed directory. Each test receives a
copied state directory; mutable stores, registries, and runtime
objects are never shared. Composition fixtures make their cost visible through
names such as `fresh_complete_runtime`, `attached_complete_runtime`, and
`authorized_complete_runtime`.

Pull-request CI runs independently selected unit, component, domain,
composition, storage, process, MCP, and e2e jobs. Provider and Lean lanes follow
the topology policy and normally defer to the exhaustive merge-queue/main gate;
`ci:lean` and `ci:full` can add them to a pull request. Domain and composition
shards may use validated timing history, while storage, process, provider, Lean,
and e2e retain separate resource lanes. Merge-queue and `main` add compatibility
and coverage; scheduled validation owns alternate ordering, stress, and
optional-provider evidence. Python 3.13 runs the small
compatibility smoke target rather than duplicating every correctness lane. A
manually dispatched debug workflow accepts one pytest node or file for focused
reproduction.

Optional providers are selected only when the production readiness probe says
their complete environment is usable (executable, version/toolchain,
libraries, and capability initialization). A missing provider removes only its
boundary capabilities.

When coverage is enabled for an exhaustive plan, each compatible Python lane
writes raw coverage data and a dependent job combines the files before enforcing
the repository threshold. The shard count and lane policy are owned by
[`.github/ci-config.json`](../../.github/ci-config.json).

For pull requests, a tested path planner reads the compiled
[`.github/ci-impact.json`](../../.github/ci-impact.json) projection and makes
independent semantic Python, Lean, npm, static, build,
security, and duplicate-code decisions. Documentation-only changes run only
the dedicated link checker; npm-only changes stay narrow. Ordinary capability
source selects its owning semantic Python lanes plus static and build, without
Lean, security, or duplicate-code by default. Verification-runtime
boundaries, packaging, CI, and unknown paths fail closed to all functional
lanes. Merge-queue checks and pushes to `main` always use the exhaustive plan,
which also enables coverage and the second Python version. Stable aggregate
Python and Lean jobs preserve required status semantics when their underlying
matrices are conditional. Maintainer-applied `ci:full` and `ci:lean` labels can
force all lanes or add Lean respectively; label events re-trigger CI so the
override applies without an extra push. Overrides are additive only and
cannot weaken the plan selected from changed paths. A scheduled validation
workflow separately exercises repeated property stress, alternate ordering
seeds, and optional providers outside the pull-request critical path.

Benchmark validation is decomposed into evidence roles even though CI shares
one checkout for the deterministic contract gate: task-local verifier/support
validation, task topology and digests, schemas and generated records, adapter
checks, host validation tests, and Oracle execution. A task README is documentation; a task
instruction, environment, manifest, or member record is executable evaluation
input. Shared environment profiles and execution-control changes may escalate
to merge-queue portfolio evidence. Ignored Python bytecode is excluded from
source validation, while committed or explicitly unignored cache files remain
invalid.

Local execution-configuration work has its own focused handoff. `make
harbor-execution-check` validates repository-wide Harbor contracts and the
unit tests that own job JSON, MCP configuration, job-level Compose overlays,
and their execution helpers. It deliberately excludes the task-specific
mathematical verifier regressions under `benchmarks/validation/`; `make
harbor-check` retains that full integration role. Task
`environment/docker-compose.yaml` files are executable benchmark input, not
job overlays, and remain gated by `make harbor-check-task` and
`make harbor-oracle-task`. Neither command starts an Oracle or model.

The build lane produces the source distribution and wheel once. Its dependent
package-validation job downloads that artifact and exercises both installed
entry points from the wheel instead of rebuilding the project. Test jobs still
use the checked-out source because their repeated cost is dependency setup and
test execution, not package compilation; CI does not cache or transfer a
relocation-sensitive virtual environment between runners.

Model-in-the-loop evaluations are not tests. Routine targets and CI may exercise
their loaders, scorers, replay paths, telemetry parsing, and dispatch guards
with deterministic fixtures, but never start an evaluated model. A human must
use the separate `make agent-eval` entry point, select cases explicitly, review
the plan, and opt into execution with `EVAL_EXECUTE=1` and a bounded model-run
count. See
[Agent evaluations](evaluations/evaluation-methods.md#workflow-observations).

## Criticality classes

Every module and change is assigned the highest applicable criticality class.
The class determines the minimum test and review evidence.

### C0 — Trust-critical

C0 code can cause false evidence to become verified, bind evidence to the wrong
mathematical object, or corrupt the identity of an object. It includes:

- canonical encoding and mathematical object digests;
- exact integer and rational parsing and normalization;
- common result-state invariants;
- artifact atomicity and digest verification;
- checker authorization, compatibility, and revocation;
- witness and certificate binding and replay;
- verification cache keys;
- the dependency boundary between search and checkers.

C0 changes require:

- an attack or invariant test written before the implementation change;
- public-interface behavior tests;
- property or state-machine tests where the input space is combinatorial;
- the relevant adversarial conformance tests;
- an independent exact-diff review;
- rerunning every affected C0 suite against the final tree.

No C0 test may be marked flaky, retried until green, or converted to an expected
failure to unblock a release. An intermittent C0 failure is a product defect.

### C1 — Correctness-critical

C1 code can misreport experimental scope, lose evidence, or make a capability
behave incorrectly without directly authorizing a verified conclusion. It
includes:

- plugin manifests and capability dispatch;
- capability discovery, invocation, and domain adapters;
- evaluation, construction, and witness-search capabilities;
- shrinking;
- budgets, cancellation, and partial batch handling;
- CLI and MCP equivalence;
- reference plugins and their search-side implementations.

C1 changes require behavior tests, boundary-focused integration tests, and
property tests where useful. A change that affects a C0 dependency is reviewed
and tested as C0.

### C2 — Supporting

C2 code includes human-facing formatting, documentation rendering, development
utilities, and reports that cannot alter stored identities or verification
outcomes. It receives ordinary unit or snapshot coverage where that protects a
public behavior. Exact private formatting is not tested unless it is a promised
wire format.

## Critical-area risk register

| Area | Catastrophic failure | Minimum release evidence |
| --- | --- | --- |
| Result semantics | Timeout, error, or incomplete search becomes a mathematical conclusion | Generated state matrix, malformed-wire tests, conformance IDs `RES-*` |
| Artifact identity | Distinct semantics collide or metadata changes mathematical identity | Canonicalization properties, domain-separation tests, independent fixture digests |
| Evidence binding | A valid witness or certificate is reused for the wrong target | Field-by-field substitution matrix, clean-process replay, `WIT-*` and `CRT-*` |
| Checker authority | A plugin or caller selects code that certifies itself | Registry state machine, administrative-boundary tests, executable-digest checks |
| Checker independence | Finder and checker share the same semantic omission | Dependency tests, deliberately separate implementation, differential fixtures |
| Storage and cache | Partial/corrupt data or stale results are accepted | Real-filesystem crash tests, digest-on-read, cache-key mutation matrix |
| Semantic completeness | Intended or sampled objects are reported as the full domain | Adversarial hidden-object fixtures, scope certificates, honest coverage fields |
| Resource limits | Exhaustion changes truth or leaves a half-committed result | Fault injection, bounded parser/store tests, mixed-batch timeout/cancellation tests |
| Shrinking | A smaller but invalid artifact replaces the verified target | Per-step checker replay, cycle/budget state machine, minimality-label attacks |
| Representation changes | A relaxation or restriction is treated as equivalence | Direction tests, proof-obligation replay, proposer/checker separation |
| Research corpus integration | Retrieved hypotheses silently gain verified status | Trust-label, provider-isolation, retention, and temporal-cutoff tests |

The first seven rows are the highest-risk areas. They should be reviewed
before optimizing throughput or expanding the public tool surface.

## Test layers

The suite uses several complementary layers. Their names describe different
kinds of evidence; they are not a ladder in which a higher layer makes the
lower ones unnecessary.

### Contract tests

Contract tests exercise versioned schemas and public Python, CLI, resource, and
MCP interfaces. They cover accepted inputs, rejected inputs, round trips,
result envelopes, compatibility behavior, and generated JSON Schema.

### Property tests

Property tests use generated data to explore invariants that example tests
cannot cover adequately. Hypothesis is the initial Python implementation.
Important targets include:

- canonical rational normalization;
- canonical encode/decode round trips;
- equivalent input encodings producing one mathematical identity;
- semantics-relevant changes producing different identities;
- evidence mutation invalidating a binding or changing replay;
- reducer acceptance preserving the checked predicate;
- state transitions never creating verification authority.

Generated examples should be represented by domain strategies and public
constructors. Tests must not reproduce the production algorithm and compare it
to itself.

### State-machine tests

Hypothesis rule-based state machines model persistent or lifecycle behavior:

- put, read, deduplicate, corrupt, and reopen an artifact store;
- authorize, use, revoke, and audit a checker;
- begin, time out, cancel, complete, and replay a run;
- propose, accept, reject, and terminate shrink steps;
- populate, miss, invalidate, and reuse caches.

Invariants are checked after every transition, not only at the end of a
generated sequence.

### Integration tests

Integration tests use real temporary filesystems, real SQLite databases, and
real process boundaries. We do not mock our own store, registry, or checker
dispatcher. A fake is appropriate only at an actual external boundary and must
record calls rather than predict internal implementation details.

### Clean-process replay tests

A completed result bundle is replayed in a fresh process with an empty
in-memory cache. These tests catch hidden global state, unrecorded dependencies,
ambient paths, and accidental imports from the search package.

### Differential tests

Trust-critical mathematical operations should have a genuinely independent
comparison where practical:

- a simple, deliberately separate witness replay implementation;
- an external canonicalizer or solver checked against a small exhaustive
  implementation;
- two serialization/parser implementations for fixed certificate formats;
- a proof-assistant kernel for later high-assurance formats.

Sharing a stable wire schema is acceptable. Importing the evaluator, witness
oracle, search canonicalizer, or solver adapter into its checker is not.
Different languages can help defense in depth, but language diversity alone is
not independence.

### Adversarial and fault-injection tests

These tests deliberately attack the trust boundary:

- malformed and ambiguous serialized values;
- wrong-claim, wrong-candidate, wrong-semantics, and wrong-scope evidence;
- corrupted blobs and interrupted commits;
- evaluator lies about arithmetic or coverage;
- solver timeout presented as infeasibility;
- plugin attempts to authorize its own checker;
- cache entries copied across evaluator, checker, scope, or semantics versions;
- output, nesting, dependency, and storage limit exhaustion.

The required result is fail-closed behavior with a precise operational status,
not a generic exception and not a mathematical conclusion.

### Performance benchmarks

Performance runs measure throughput, latency, memory, and storage amplification
on already-correct behavior. Their design is specified in
[Performance benchmarks](evaluations/evaluation-methods.md).

### Model-in-the-loop evaluations

Model evaluations ask whether access to Jacobian improves a model's ability to
notice semantic gaps, use witnesses, preserve uncertainty, and hand off
replayable evidence. Their design is specified in
[Agent evaluations](evaluations/evaluation-methods.md).

## Component test matrix

### Schemas and common results

Required examples and properties:

- accept every valid enum combination used by the public contract;
- reject noncanonical rationals, zero denominators, JSON floats in exact
  objects, duplicate keys, excessive nesting, and oversized integers;
- normalize signs, common factors, and zero exactly once;
- round-trip every versioned schema through canonical bytes;
- preserve unknown data only where the compatibility policy explicitly permits
  it;
- reject `TIMEOUT`, `CANCELLED`, or `ERROR` records that claim a verified
  mathematical conclusion;
- reject `input.status = REJECTED` records that carry a verified conclusion;
- reject `verification = VERIFIED` without an authorized checker identity and
  supported assurance method;
- demonstrate that an exact direct witness can be verified with
  `coverage = NOT_APPLICABLE`;
- demonstrate that exhaustive floating-point evaluation remains unverified.

The state cross-product should be generated from the schema rules rather than
maintained as a few hand-picked examples.

### Artifact identity and storage

Required examples, properties, and state sequences:

- equal canonical mathematical objects have the same object digest;
- different schema, semantics, canonicalizer, or object-format versions change
  identity even when payload bytes are equal;
- manifests and run metadata can change without changing mathematical object
  identity;
- repeated and concurrent puts are idempotent;
- staging files never become addressable artifacts before commit;
- simulated process failure before and after rename leaves either the old
  complete state or the new complete state;
- digest verification catches modified and truncated blobs;
- SQLite rollback, reopen, and WAL recovery preserve manifest/blob agreement;
- unresolved, cyclic, too-deep, and too-wide dependency graphs fail within
  configured bounds;
- resource URIs cannot escape the store through path traversal, symlinks, or
  crafted digests;
- quotas reject new data without damaging existing artifacts;
- garbage collection preserves every configured root and reachable object;
- a cache cannot return a result after any semantics-relevant key component
  changes.

The state-machine oracle models the logical set of committed objects. It does
not duplicate the store's file layout.

### Plugin manifests and capability dispatch

Required tests:

- plugins may implement only the capabilities they need;
- missing or incompatible capabilities fail before execution;
- capability resolution is deterministic and version-aware;
- manifests bind their semantics and implementation digests;
- a changed implementation cannot retain the old manifest identity;
- discovery measures a package without importing its code;
- registry snapshots bind all declared capabilities to the whole-package source
  digest, runtime/build identity, and platform compatibility;
- path traversal, symlinks, bytecode-only or native module execution, and
  changed package bytes fail closed;
- domain-specific graph, matrix, solver, or proof types never enter the core
  manifest;
- a manifest cannot authorize, replace, or revoke a checker;
- two structurally different plugins use the same generic dispatch path;
- a disposable synthetic third plugin passes success, declared-failure,
  malformed-output, timeout, package-attack, and unsupported-promotion checks
  without runtime or MCP changes;
- each conformance-suite run uses fresh invocation identities and executes the
  plugin again rather than reusing prior durable results.

### Checker registry

Required examples and state sequences:

- only the operator administration surface can authorize a checker;
- authorization binds executable digest, supported schemas, semantics,
  evidence formats, and version ranges;
- callers cannot override checker selection through certificate input;
- duplicate identifiers with different executable digests are rejected;
- incompatible checkers are rejected before evidence replay;
- revocation blocks new verification records while preserving the historical
  identity and policy state of old records;
- checker replacement creates a new identity and invalidates verification
  cache hits;
- authorization and revocation are auditable and transactionally durable;
- time-of-check/time-of-use substitution of a checker executable is detected.

### Witness verification

Required tests:

- a valid witness is checked for domain membership and logical effect;
- an otherwise valid witness bound to another claim, semantics, candidate, or
  role is rejected;
- deletion, insertion, permutation, and value mutation of witness components
  either changes the checked meaning or causes rejection;
- malformed or over-limit witnesses fail before mathematical replay;
- a witness remains independently verifiable when the oracle is unavailable;
- a witness found after an oracle timeout is judged only by replay, never by
  the oracle's status;
- checker code cannot import evaluator, oracle, mutator, or search-solver
  packages;
- the two reference plugins use different witness shapes and replay logic.

### Certificate verification

Required tests:

- certificate format and version select an operator-authorized checker;
- certificate payloads self-describe their format and bind claim, semantics,
  candidate when applicable, encoding when applicable, and payload;
- substitution of any bound digest fails before replay;
- direct finite witness and complete finite enumeration formats have
  independent clean-process replay tests;
- malformed, truncated, unsupported, over-limit, and revoked-checker
  certificates remain unverified;
- a solver's `optimal`, `unsat`, or `complete` label is not accepted in place
  of a supported checked certificate;
- verification records are immutable and identify the exact checker digest;
- repeated verification may use a cache only when all bindings and policy
  inputs match;
- parser and replay failures return an operational error or rejected evidence,
  never a mathematical falsehood.

Later certificate formats add format-specific mutation, differential, and
known-answer suites before authorization is possible.

### Evaluation, construction, and witness search

Required tests:

- a batch may contain accepted, rejected, completed, timed-out, and errored
  candidates without one result contaminating another;
- returned results preserve input ordering or carry stable candidate identity;
- actual arithmetic, coverage, scope, limits, evaluator digest, seed, and
  environment are recorded;
- reaching any search or enumeration limit prevents exhaustive coverage;
- heuristic, sampled, restricted, and exact evaluation remain unverified;
- proposed evidence is stored by URI rather than embedded without bounds;
- a timeout, cancellation, crash, or partial backend response never becomes
  `FALSE`, `INFEASIBLE`, `NONE_CERTIFIED`, or `VERIFIED`;
- `NONE_CERTIFIED` is returned only with a successful verification record bound
  to the exact search question;
- changing evaluator, scope, limits, seed when semantically relevant, or
  environment invalidates the corresponding cache entry;
- evaluator and oracle bugs cannot write to the checker registry.

### Capability composition

Multi-step mathematical strategies remain agent-owned. Tests therefore exercise
the artifacts and contracts at capability boundaries rather than blessing one
prescribed workflow:

- every invocation records the exact descriptor, inputs, mode, budgets, backend
  identity, outputs, and parent artifacts used;
- outputs from one capability can be passed to another without losing type,
  provenance, scope, completeness, or assurance;
- intermediate failures, rejected candidates, and proof obligations remain
  inspectable rather than disappearing inside a composite result;
- repeated, reordered, or abandoned invocations cannot strengthen assurance;
- search, generation, transformation, ranking, and retrieval outputs remain
  unverified until an authorized independent checker accepts bound evidence;
- a verified record from one claim, semantics, candidate, scope, or checker
  version cannot be reused for another;
- Python, CLI, and MCP invocation paths preserve the same capability contract
  and verification boundary.

### Shrinking

Required examples and state sequences:

- every accepted reduction strictly improves at least one declared objective
  without worsening a higher-priority objective under the selected ordering;
- every accepted reduction receives a fresh preservation-checker replay;
- a reduction that breaks the predicate is rejected without changing the
  current target;
- a malformed or nonterminating reducer is bounded;
- repeated candidates and cycles do not cause nontermination;
- budget exhaustion reports the strongest minimality actually established;
- the current contract rejects `ONE_STEP`, `BOUNDED_GLOBAL`, and
  `PROVED_GLOBAL`; no supported contract may emit them without the
  corresponding checked completeness evidence;
- the final output, accepted-step trace, rejected proposals, objectives, and
  checker identities can be replayed;
- candidate and witness targets use typed reducers and cannot be confused;
- certificate simplification requires a separate capability contract.

### CLI and MCP capability surface

The CLI and MCP layer must be thin enough to test by equivalence:

- the Python API, CLI, and `math.run` return the same semantic result
  envelope for the same descriptor version and artifact inputs;
- `math.find`, `capability://catalog`, and invocation schemas agree;
- boundary tests assert the exact tool, resource, template, and prompt inventories,
  their safety annotations and schemas, the operating guide, and representative
  browse, query, and exact-description behavior through their public MCP seams;
- malformed requests fail before runtime invocation;
- large artifacts and traces are returned as resource URIs;
- response-size limits are enforced;
- cancellation and progress never mutate mathematical conclusions;
- stdio startup and one request work in a clean installed environment;
- adapter packages contain no domain mathematics or verification policy.

### Reference plugins

The public mathematical inputs and expected oracles are selected in the
[Mathematical scenario catalog](scenarios/math-scenarios.md).

The finite directed-graph/path reference plugin must contain:

- a candidate whose intended family omits a legal induced object;
- an oracle that returns the unexpected path;
- a separately implemented checker that accepts the witness;
- an exact, verified structural counterexample and finite certificate;
- deliberately corrupted variants for every binding dimension.

The second reference plugin must use a non-graph candidate, a different witness
shape, and different optional capabilities. Its exact statement is frozen
before implementation so it cannot be chosen merely because the current core
already happens to pass it.

## Test organization

Directory ownership is the source of truth:

```text
tests/
    unit/
    component/
    domain/
    composition/
    boundary/
    e2e/
    support/
```

Package-local tests are appropriate for focused behavior. Cross-package trust
tests live in the top-level groups above. Runtime pytest markers are limited to
execution-affecting traits:

```text
requires_provider(name)
performance
property
destructive_process
```

Conformance and differential are mathematical evidence categories, not pytest
execution markers. Place those cases under their owning unit, component,
domain, boundary, or e2e directory. Directory ownership and the topology
manifest select tests; markers must not become a second suite taxonomy.

Use real clocks only when time itself is under test. Otherwise inject a clock,
random source, executor, or backend at a public seam. Use temporary real SQLite
databases and filesystems rather than mocks of persistence behavior.

## Test tooling

The initial test stack remains deliberately small:

| Need | Initial tool | Use |
| --- | --- | --- |
| Test runner and fixtures | pytest | Public behavior, semantic lanes, boundaries, and conformance suites |
| Generative and stateful testing | Hypothesis | Canonicalization, parser, persistence, lifecycle, and reduction invariants |
| Language-neutral schema validation | `jsonschema` | Check generated wire contracts independently of Pydantic parsing |
| Coverage diagnostics | coverage.py through pytest-cov | Find unexercised code; never substitute percentage for behavior coverage |
| Performance measurement | pyperf | Calibrated Python benchmarks with metadata and raw JSON |
| Static checks | Ruff plus a strict Python type checker selected during scaffolding | Catch ordinary defects before runtime suites |

The checked-in JSON Schemas should be exercised both through Pydantic and a
standards-oriented JSON Schema validator. A Pydantic model successfully reading
its own generated schema is not independent contract evidence.

### Pytest antipatterns

- Use `monkeypatch` for process-global state such as `sys.modules`, `sys.argv`,
  and environment variables. When restoration is the behavior under test,
  assert that the original object or missing state is restored.
- Name the expected exception type and a stable diagnostic instead of using
  bare `pytest.raises(Exception)`.
- Assert the observable outcome or independently parsed wire contract; a model
  successfully validating its own output is not sufficient evidence.
- Name source-shape checks honestly and reserve them for supported text or
  architecture contracts. Do not disguise source substrings as behavior tests.
- Do not add an empty `pytestmark`; markers exist only for execution-affecting
  traits owned by the test topology.
- Keep Harbor regressions in their owning dataset and task leaf. Shared
  validation modules should contain only suite-wide contracts.

Use the standard library's `tempfile`, `subprocess`, and process primitives for
artifact and replay tests. Do not use an in-memory filesystem or `pyfakefs` to
claim evidence about atomic rename, SQLite WAL recovery, symlink handling, or
durability. Those behaviors require a real filesystem.

Run the affected tests sequentially while developing lifecycle and concurrency
behavior, then use the bounded xdist lane for the full suite. Parallelizing a
test suite can obscure shared-state bugs; concurrency is introduced through
explicit state-machine and integration scenarios first. Similarly,
select a mutation-testing tool only after the C0 suite is stable enough that
surviving mutants are actionable rather than noise.

## Test areas

Testing belongs to each implementation change rather than to one cleanup issue
at the end. Cross-cutting harness work is organized into these concurrent
areas:

### Test foundation

- configure pytest markers and bounded Hypothesis profiles;
- add canonical fixture/artifact builders through public constructors;
- add real temporary store and clean-process helpers;
- add checked-in schema validation through `jsonschema`;
- make random seeds and failure artifacts visible in CI output.

### Result and artifact invariants

- implement generated result-state cross-product tests;
- implement rational and canonical-encoding properties;
- implement artifact-store state machines and crash points;
- cover cache-key domain separation.

This work ships with the schema and artifact issues, not after them.

### Trust-boundary conformance

- make the conformance IDs executable and traceable to tests;
- implement checker-registry state machines;
- implement evidence-substitution mutation matrices;
- enforce package/import and clean-process boundaries;
- add direct-witness and finite-enumeration known-answer fixtures.

### Capabilities and shrinking

- add mixed-batch and resource-failure scenarios;
- add oracle outcome and `NONE_CERTIFIED` protocol tests;
- add reduction state machines, cycle detection, and minimality-label tests;
- compare Python API, CLI, and `math.run` results.

### Cross-domain fixtures

- freeze the two reference problem statements and hidden expected facts;
- implement independent search and checker paths;
- add adversarial variants and replayable bundles;
- demonstrate that neither plugin changes core schemas.

### Performance baseline

- freeze the benchmark corpus;
- add pyperf commands and raw-result artifact collection;
- establish same-host variance before proposing regression gates;
- profile only after correctness and baseline results exist.

### Portfolio-evaluation harness

- isolate public fixtures from hidden oracles;
- implement same-model, same-budget baseline and portfolio conditions;
- record every run, seed, model/tool version, and hard failure;
- include hidden independent oracles and held-out cross-domain cases;
- record correctness, false certification, runtime, tokens, calls, and
  parameter errors;
- pilot semantic-closure, timeout, binding, shrinking, and composition cases.

These areas are concurrent, not a release ladder. Trust-boundary
coverage remains mandatory whenever a capability can affect verification.

## TDD implementation sequence

The test plan is an inventory, not an instruction to write thousands of tests
before implementation. Each issue proceeds in thin vertical slices:

1. Add one failing public behavior or attack test.
2. Confirm it fails for the intended reason.
3. Implement the smallest contract-preserving behavior.
4. Run the focused suite.
5. Refactor only while the behavior remains green.
6. Add the next boundary case.

The first slices should be:

1. Reject an operational timeout carrying a false verified conclusion.
2. Canonicalize equivalent rationals and reject ambiguous exact values.
3. Give equal objects equal digests and domain-separated objects different
   digests.
4. Recover atomically from an interrupted artifact put.
5. Reject a plugin attempt to register a checker.
6. Reject a valid certificate rebound to another candidate.
7. Accept and replay one direct witness in a clean process.
8. Reject a shrink step that is smaller but does not preserve the predicate.

This sequence crosses the result, storage, registry, verification, and
orchestration boundaries early enough to expose a flawed architecture before
large modules accumulate.

## CI and release gates

### Local and change-focused

During implementation, run the smallest affected contract, property, or
integration subset. Static formatting, lint, and type checks run with it once
their configuration exists.

### Pull request

Every pull request runs:

- deterministic contract tests;
- bounded property and state-machine profiles;
- real SQLite/filesystem integration tests;
- applicable subprocess, conformance, and dependency-boundary tests;
- both reference-plugin tests once they exist.

C0 changes additionally require an independent exact-diff review. After any
resulting edit, only invalidated focused evidence is rerun, followed by one full
required validation pass against the final tree.

### Nightly

Nightly validation runs:

- higher Hypothesis example counts and longer state-machine sequences;
- fault-injection and crash-recovery matrices;
- differential implementations and optional external backends;
- clean-install and clean-process replay;
- performance measurements on a controlled runner;
- mutation testing pilots for selected C0 modules.

Nightly failures are triaged as defects. Retrying is useful for diagnosis, not
for declaring the original result green.

### Release

A release requires:

- every required contract and fail-closed verification test;
- both structurally different reference plugins;
- package installation and replay in a clean environment;
- synchronized Python and npm versions with both packages built and tested;
- no unresolved C0 failure or accepted flake;
- a replayable artifact bundle for each reference result;
- recorded checker identities and dependency versions;
- an independent review of final C0 diffs since the preceding release.

Code coverage is diagnostic, not a release proxy. Completion of the behavior
and attack matrix is the primary measure. Mutation testing can strengthen C0
confidence after the suite is stable, but a mutation score is not itself a
mathematical assurance level.

## Bounded-discovery coverage

- enumeration scope, page-progress, limit, and cancellation behavior;
- isomorphism/canonicalization differential tests;
- canonical keys bound to canonicalizer implementation identity;
- transformation direction and proof-obligation tests;
- exact projection and separator replay;
- capability budget, cancellation, progress, and artifact-replay behavior.

The critical-area register and domain verifier tests define the attack matrix.
Complete search snapshots remain unverified; a theorem inferred from the
absence of a candidate needs a domain-specific completeness certificate and
checker.

## Capability portfolio coverage

Capability availability is not a compatibility or verification claim. Every
installed capability needs descriptor, schema, invocation, limit, cancellation,
artifact, and provenance tests appropriate to its semantics. Experimental
capabilities may change contracts, but they must still fail closed and must not
obtain checker authority.

Portfolio evaluations complement contract tests. Run agents with the full
portfolio and with controlled ablations using the same model, budget, seeds,
task inputs, and hidden oracle. Include held-out tasks beyond the domains used
to design the capabilities. Record correctness, false certification, runtime,
tokens, calls, and parameter errors. Use transcripts to improve discovery,
examples, routing, batching, and capability boundaries; do not treat prescribed
tool use as evidence that the portfolio helps autonomous agents.

Provider-backed retrieval additionally requires trust-label preservation,
provider isolation, retention and quota state machines, temporal cutoffs, and
data-leakage tests. A compromised or unavailable provider cannot mutate
artifacts, checker authority, or the verification boundary.
