# Testing strategy

[Documentation home](../index.md)

Jacobian tests observable mathematical behavior, typed boundaries, execution
semantics, publication, and independent checker authority. The suite does not
encode a required research sequence.

## Change matrix

| Change | First local check | Escalate when |
| --- | --- | --- |
| Documentation | `make docs-linkcheck` | Checked examples also run when their supported contract changes |
| Python behavior | `make check`, then the owning named lane | Use `make check-all` only for an intentional complete ordinary reproduction |
| Mathematical domain | Owning domain/component tests | Add composition only for real cross-domain handoff |
| MCP projection | Focused MCP boundary tests | Add stdio/HTTP parity when transport behavior changes |
| Checker protocol or authority | Focused checker and authority tests | Require an independent exact-diff review |
| Storage, hosting, provider, or process boundary | Named specialist lane | Add crash/restart or remote evidence only when that boundary changes |
| Harbor task input or verifier | `make harbor-validate-task DATASET=... TASKS=...` | Run exact Oracle after executable task/verifier changes |
| Deployment entrypoint | `make deploy-check` | Include affected process checks for code changes |

Before final validation, run the ordinary check on the frozen tree:

```sh
make setup
make check
```

Default `uv run pytest` collects the Lean-free ordinary `testpaths`. Use the
Make targets that own storage, process, MCP, and Lean isolation. `make
check-external` covers the pinned Lean specialist lane when that tree changes.

`make check` is the bounded local handoff: lint, typecheck, and unit tests.
Hosted CI runs the complete ordinary suite as six fixed semantic lanes: `unit`,
`component`, `domain`, `composition`, `e2e`, and `provider`. These are static
Make targets, not path-selected or timing-planned shards. Use the named lane
that owns the change; `make check-all` runs all six locally in the same order.

## Test ownership

The filesystem is the metadata. A test under `tests/domain/` is a domain test;
Lean lives under `tests/boundary/providers/lean/`. Directory prefixes are
exclusive (longest prefix wins).

| Directory/lane | Evidence owner |
| --- | --- |
| `tests/unit` | Pure values, contracts, parsers, and small policies |
| `tests/component` | One real service, adapter, operation binding, or provider seam |
| `tests/domain` | One explicitly selected mathematical domain |
| `tests/composition` | Cross-domain handoff, complete installation, checker authority, lifecycle |
| `tests/boundary/storage` | Persistence, transactions, crash/restart, bounded reads |
| `tests/boundary/process` | Worker identity, cancellation, resource enforcement |
| `tests/boundary/mcp` | SDK schema, structured output, resources, stdio/HTTP behavior |
| `tests/boundary/providers` | Optional provider readiness and identity |
| `tests/boundary/providers/lean` | Pinned Lean/Mathlib boundary |
| `tests/e2e` | Complete caller-visible journeys |

Ordinary lanes invoke pytest directly. Three lanes wrap pytest in
`tools/pytest_lifecycle.py` because child process trees can wedge: process,
MCP, and Lean.

Use the narrowest production graph that proves the assertion. Complete-runtime
fixtures are reserved for complete inventory, cross-domain wiring, checker
authorization, lifecycle, and host boundaries. One operation's request/result
matrix belongs to its domain or component seam.

Admission follows the asserted contract:

- a test that names one mathematical domain belongs under that domain;
- a test that directly validates a model belongs in unit;
- one adapter, checker, or provider seam belongs in component;
- composition requires a cross-domain edge, complete-portfolio invariant,
  authority transition, hydration lifecycle, or global policy;
- boundary tests require the real SQLite, process, MCP, Lean, or optional-provider
  boundary they assert; and
- e2e retains only complete caller journeys that would be materially weaker when
  decomposed.

Names such as `frontier`, `migration`, `regression`, `release`, and issue numbers
describe history rather than ownership. Put a regression under its permanent
semantic owner and give it a descriptive behavioral name.

The canonical commands are:

```sh
make test-unit
make test-component TESTS=tests/component/providers/public_api/test_matrices.py
make test-domain TESTS=tests/domain/polynomial/test_polynomial_operation_bundle.py
make test-composition
make test-storage
make test-process
make test-mcp
make test-provider
make test-lean
make test-e2e
make test-exhaustive
make test-checker-subprocess-coverage
make quick
make check
make check-all
make check-external
make check-static
```

`make test-all-ci` is an explicit exhaustive reproduction, not a routine gate.
Before starting it, confirm that no other pytest job from this checkout is
running. Concurrent runtime/store/subprocess suites can turn per-test
timeouts into host-contention noise. Exhaustive local targets
(`test-all-ci`, `test-exhaustive`, `harbor-check-all`,
`harbor-host-validation`, `harbor-oracle-all`) take an OS-locked lease in
this worktree; focused `make test-unit` stays free. `make validation-status`
reports whether the lease is held.

GitHub Actions identity is the YAML under `.github/workflows` on the default
branch. Registrations whose files are gone, including historical
`agent-port-*` and `agent-rebase-*` leftovers, are historical: disable them in
the GitHub UI and retain their run history.
`python tools/inventory_github_workflows.py` compares files to registrations
and never disables workflows. `python tools/restack_feature_branch.py` reports
unique feature commits and duplicate subjects against a declared parent; it
never force-pushes.

Ordinary CI coverage instruments each pytest process but does not automatically
instrument every child it launches. Independent checker calls therefore retain
their real fresh-process behavior without each short-lived worker producing a
coverage database. `make test-checker-subprocess-coverage` is the focused
coverage-transport contract: it enables coverage.py's subprocess patch in an
isolated profile, exercises accepted, rejected, malformed, and undeclared-import
worker outcomes, and fails unless child execution is present in the combined
data. The required aggregate coverage job includes that focused database and
retains the repository threshold.

Broad finite reference sweeps that are valuable but disproportionate for pull
requests use the `exhaustive` marker. The component lane excludes them, and
scheduled validation owns `make test-exhaustive`. Keep a representative
behavioral case in the ordinary owning lane; do not use the marker to defer
boundary, authorization, persistence, or public-API coverage.

## Test principles

Tests exercise public behavior and stable artifacts rather than private helper
names, implementation branches, or source text. A regression test must fail on
the base tree for the intended reason when that proof is feasible.

Pydantic request parsing is tested with accepted and rejected complete models,
including unknown fields and cross-field contradictions. Provider/subprocess
output receives separate malformed-output tests because it crosses a second
untrusted boundary.

Use property tests for canonicalization, round trips, normalization,
carrier invariance, and algebraic invariants when a property expresses the
contract better than examples. Use representative storage and process seams
when the claim depends on SQLite, filesystem publication, subprocesses, or
cancellation.

Let pytest own collection and fixture lifetime through the narrowest owning
`conftest.py`. Broaden fixture scope only for reusable installation state whose
tests do not mutate it; tests that revoke authority, patch shared services, or
otherwise change runtime state retain an isolated fixture.

For repeated installation, first select the smallest production domain bundle.
Group read-only assertions around an immutable module-scoped fixture only when
isolation permits it. If measurement still justifies reuse for mutating tests,
clone a quiesced state template into a private test directory. Keep a
function-scoped cold start when construction, authorization, or hydration is the
contract. Do not introduce a universal fixture registry or installation-plan
abstraction for test convenience.

Do not substitute source-reading tests for caller-visible behavior. If a
behavioral regression proof is infeasible, state the proof gap.

## Documentation acceptance

- no current-product page teaches a workflow engine, managed Skill, plugin
  emulation, compatibility aliases, or workflow resources;
- the README, `AGENTS.md`, product blueprint, and architecture express the same
  dependency direction;
- examples pass complete request parsing and run where the documented provider
  is part of the supported environment;
- public Python API documentation matches explicit `__all__` values;
- links pass; and
- capability membership remains sourced from the live catalog.

## `jacobian.math` acceptance

For each public domain:

- Python callers construct no runtime, catalog, store, or MCP object;
- installed operations call the same public semantic implementation;
- every public function has one canonical semantic input type;
- provider conversion stays private and backend imports are lazy;
- request parsing and result serialization each occur once;
- no internal `CapabilityRequest` or `CapabilityResult` call remains;
- constructor/backend and large-exact-value round trips are covered where the
  contract permits them; and
- Import Linter prevents upward dependencies.

Import Linter prevents upward dependencies. Value modules cannot import private
provider adapters.

## Operation execution acceptance

Tests cover the complete pipeline:

```text
parse once → preflight → execute → postcondition → terminal state → publication
```

Required cases include:

- preflight refusal before allocation or publication;
- postcondition failure before any value/reference/artifact exposure;
- ordinary typed results without generic scope, completeness, relationship, or
  obligation decoration;
- one serialization of a completed output;
- timeout, cancellation, provider error, and resource refusal as
  non-conclusions;
- bounded operations whose typed result distinguishes exact from incomplete or
  unknown outcomes without generic completeness or obligation decoration;
- interrupted bounded operations publish no partial result or artifact;
- publication that preserves semantic value identity;
- invocation provenance distinct from mathematical value and verification
  record identity; and
- selected-operation effect metadata preserved through catalog and run.

## Composition acceptance

Composition tests use independent producer and consumer operations. They cover:

- producer-to-consumer compatibility without JSON rewriting;
- exact parent, presentation, axis, basis, label, and version matching;
- rejection of unsafe same-shape substitutions;
- opaque reference session/tenant/lifetime checks;
- inline/reference/artifact carrier invariance;
- multi-input completeness—a checker missing one required input is not
  invocable;
- explicit transforms rather than hidden coercion; and
- no assurance propagation through compatibility or carriers.

Installation tests validate supported port accessors against their declared
value type. Finite-field linear algebra and polynomial maps use the same
whole-value input and output ports; tests reject any declaration whose accessor
disagrees with the Pydantic request or result type.

## Finite-field composition acceptance

Direction-ledger tests bind the exact finite-field modulus, generator, ordered
basis, encoding version, and presentation digest. They reject differently
presented isomorphic fields and wrong axes/bases, construct the explicit
`F₂⁴ → F₂⁶` restricted-scalar map, agree on rank between Python-FLINT and the
independent prime-field path, preserve all nine projective directions in the
rank ledger, and independently replay the final orbit distribution.

Adversarial cases mutate presentation, axes, basis, rank, direction identity,
and ledger bindings. Cost admission occurs before allocation/publication and no
input assurance is inherited.

Polynomial-map tests reuse the same field presentation, element values, codec,
carriers, and references. A finite map table binds exact domain and codomain;
fibers partition the complete exact field; collision and permutation
certificates bind the exact map and enumeration scope; forged or incomplete
tables fail; producer and checker share no executable enumeration.

## MCP and CLI acceptance

MCP tests assert exactly two statically registered tools, generated schemas,
strict extra-argument rejection, Pydantic structured output, `ResourceLink`
only when needed, and stdio/HTTP parity. The client journey covers:

- search and exact inspection;
- valid run;
- unknown top-level argument;
- invalid selected payload;
- provider unavailable;
- producer-to-consumer value reference;
- multi-input checker;
- producer-to-checker handoff;
- durable resource link and read; and
- cancellation and packaged connector identity.

CLI tests prove that `catalog`, `inspect`, and `run` use the same installed
specifications, preflight, execution, and publication semantics as MCP.

## Checker and infrastructure acceptance

Checker tests separate availability from authorization, bind every verification
record field, and prove that timeout/interruption cannot commit. Per-checker
identity tamper tests cover source closure, passive contracts, worker runtime,
Python identity, dependency files, native libraries/executables, and sandbox
policy. Unrelated product changes must not alter checker identity; actual
checker dependencies must.

The completed storage experiment retained the filesystem CAS. Its disposable
SQLite comparison remains reproducible evidence rather than a selectable
backend. Storage regression tests therefore focus on the retained CAS's
transaction coordination, quota recovery, bounded reads, crash/restart, and
backup/restore behavior.

Run the disposable carrier comparison with:

```sh
uv run python tools/benchmark_storage_blobs.py --iterations 16 \
  --output storage-blob-benchmark.json
```

The spike is historical decision evidence, not a selectable runtime backend or
a second production abstraction. Re-run it only when new workload evidence
justifies reopening the storage decision.

Hosting tests keep local and remote ownership separate. Remote authentication,
tenants, admission, leases, eviction, and quarantine must not enter the local
server. Provider tests prove that importing `jacobian.math` performs no probe,
private provider imports are lazy, and absence removes only affected catalog
entries.

## Model evaluations

Model-in-the-loop evaluations are explicit operator-run evidence, never a
routine development gate. The control has no Jacobian; the treatment has
Jacobian MCP only. Hold model, prompt, budget, dataset revision, and environment
constant. Score mathematical correctness, useful intermediate values, safety,
and efficiency without rewarding a prescribed tool-call order.

Harbor tasks, hidden verifiers, and Oracle runs remain evaluation infrastructure
outside the runtime product. Follow the repository-local `harbor-benchmarks`
skill and exact task validation path when those files change.

## Final validation

After implementation freezes the behavioral tree:

1. run `make check` and the named lane that owns the change;
2. run any named specialist lane the change actually crossed;
3. complete any required independent exact-diff review;
4. resolve its consolidated findings;
5. rerun the invalidated focused checks on the final tree; and
6. report only evidence that actually ran, including material proof gaps.

After any edit, rerun only checks whose evidence the edit invalidated. Do not
repeat expensive full, race, integration, cross-platform, or provider suites
when the relevant tree is unchanged.
