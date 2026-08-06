# Contributing to Jacobian

Jacobian is a pre-stable 0.6.0 capability workbench. It exposes composable mathematical
capabilities that AI agents use to investigate conjectures and other
mathematically specified problems. Contributions should preserve mathematically
atomic, agent-visible outcomes, agent-owned composition, and the boundary
between heuristic search or evaluation and independently verified evidence.

## Before changing code

Read the [documentation home](docs/index.md), the
[product goals](docs/explanation/goals.md), and the
[testing strategy](docs/reference/testing-strategy.md).
Use the installed catalog and current reference documents for present
capability membership.

## Validation at a glance

| Change | Start here | Oracle requirement |
| --- | --- | --- |
| Documentation, benchmark README, or `benchmarks/validation/` | `make docs-linkcheck` | None |
| Focused Python behavior | `make test-plan BASE=<revision>`, the selected lane, then `make check` | None |
| Harbor job JSON, MCP config, job-level Compose overlay, or execution helper | `make harbor-execution-check` | None |
| Benchmark task input or verifier | `make harbor-check-task DATASET=... TASKS=...`, then `make harbor-oracle-task DATASET=... TASKS=...` | Exact selected-task Oracle |
| Deployment entrypoint | `make deploy-check` | None |
| CI, dependencies, or unknown paths | `make check-static` plus affected tests | As required by the selected plan |

The four primary test profiles are the cheapest useful starting points:
`unit` covers pure contracts and models; `component` exercises one real service
or adapter; `domain` loads explicitly selected mathematical bundles; and
`composition` checks complete runtime wiring. Use the named boundary profiles
(`storage`, `process`, `mcp`, `provider`, `lean`, and `e2e`) when the change
crosses one of those boundaries. The [testing strategy](docs/reference/testing-strategy.md)
defines the ownership and escalation rules.

## Development environment

Jacobian uses Python 3.12, the uv release pinned in `.uv-version`, and a small
`Makefile` that keeps local commands aligned:

```sh
make setup
make test-unit
make test-component
```

The suite is split into semantic lanes. Use the narrowest lane that proves the
behavior: `unit` for pure contracts and models, `component` for one real
service, `domain` for explicitly installed mathematical bundles,
`composition` for complete runtime wiring, and the `storage`, `process`,
`mcp`, `provider`, `lean`, and `e2e` lanes for their named boundaries. Each
semantic lane target accepts a pytest file or node through
`TESTS=<file-or-node>` and extra pytest options through `PYTEST_ARGS`.
`make check` runs Ruff, mypy, and the
unit lane; it is a useful local handoff, but the pre-push hook intentionally
runs only `make lint typecheck` so it stays below the interactive feedback
budget. CI owns path-planned correctness lanes and optional environments.

`make check-changed` combines the static edit-loop checks with exact changed
test selection. `make check-static` adds dependency/dead-code checks and a
package build when a focused change needs them. `make test-all-ci` is the explicit exceptional local
reproduction of every semantic lane. Run `make help` for the complete command
index of common commands; use `make help-all` for lifecycle and diagnostic
plumbing. Lane ownership and local commands are documented in
[testing strategy](docs/reference/testing-strategy.md).

Use `make test-plan BASE=origin/main` for exact local test selectors and
`make ci-plan BASE=origin/main` for the broader hosted-CI lane decision plus a
provenance-bound plan receipt. The two reports intentionally answer different
questions: the local plan may run exact importing tests, while the hosted plan
owns required semantic lanes and fail-closed infrastructure coverage. Use
`make harbor-plan BASE=origin/main`
for benchmark contracts and Oracle scope; run it through Make because the
planner requires the pinned Harbor runtime to compute task digests.
The plan also selects host-side verifier regressions: task- or dataset-owned
modules for focused changes, the changed validation file itself, and the full
`benchmarks/validation` suite, sharded in hosted CI, for shared or unclassified
infrastructure. To reproduce one selected host check locally, run
`make harbor-validation-tests TESTS=<pytest-file-or-directory>`.
Tests can be narrowed without learning another wrapper:

```sh
make test-component TESTS=tests/component/capabilities/test_atomic_capabilities.py
make test-component TESTS=tests/component/capabilities/test_atomic_capabilities.py PYTEST_ARGS="-k schema -n 0"
make test-unit TESTS=tests/unit/contracts/test_result_envelope.py
make test-process TESTS=tests/boundary/process/search/test_shrinking.py
make test-mcp PYTEST_ARGS="-k authentication"
make test-storage TESTS=tests/boundary/storage/transactions/test_state_database_migrations.py
make test-lean TESTS=tests/boundary/providers/lean/test_lean_repl_runtime.py PYTEST_ARGS="-k induction"
```

All Makefile pytest targets print their ten slowest tests by default. Set
`PYTEST_DIAGNOSTIC_ARGS=--durations=0` to suppress that report, or use a larger
value such as `PYTEST_DIAGNOSTIC_ARGS=--durations=25` while investigating a
regression.

Run `make hooks` once to install commit-time formatting, syntax, secret,
large-file, dead-code, and actionlint hooks plus the static `make lint typecheck`
pre-push gate. Use `make check` when you also want the unit lane locally. Hooks
remain bypassable for exceptional cases with Git's standard `--no-verify`
option.
`make fix` applies Ruff's safe lint fixes followed by formatting. `make
precommit` applies those fixes and then runs the routine handoff checks.

On macOS, read the
[Z3 installation guide](docs/how-to/troubleshoot-z3-macos.md) before troubleshooting a
source-build failure from `uv sync --dev`.

Use focused tests while implementing. Run `make check` (or the affected
semantic target) and wait for green CI checks before merge. Run broad local
validation only when changing CI itself, debugging an environment-specific
failure, or when CI is unavailable. `make test-all-ci` is an explicit exception
path, not a routine confidence gate. Before it, verify that no other pytest or
delegated-agent validation is running on the host. Never assign an exhaustive
suite to a parallel agent sharing the checkout. Report only checks that
actually ran. The manually
dispatched Python Debug and Lean Debug workflows reproduce one pytest file or
node in a prepared remote environment when the relevant local runtime is
impractical.

CI classifies pull requests through the tested source-to-suite impact
manifest in `.github/ci-impact.json`. Documentation-only changes skip
Python, npm, Lean, static, package, security, and duplicate-code lanes, but
run the dedicated `make docs-linkcheck` lane.
Documentation plus npm or npm-only changes run npm packaging without the
Python and Lean lanes. Unknown paths fail closed to all functional lanes.
Required status contexts still complete after checking the plan when their
expensive validation is intentionally omitted.
Maintainers can add the `ci:full` label to force every lane or `ci:lean` to
add real-Lean validation to an otherwise isolated plan. Label changes re-trigger
CI so the override applies without an extra push. These overrides only add work;
labels cannot reduce the fail-closed path classification.
Pull requests run the canonical Python version. Merge-queue groups and pushes
to `main` additionally run supported-version compatibility and combined
coverage as exhaustive gates. Successful `main` runs publish fresh integration
shard timings. Timing history is not committed, and missing or invalid history
falls back to equal-weight sharding.

Use `make test-plan BASE=<revision>` to preview the same changed-path routing
before validation. For an added or modified leaf Python module, the planner
selects test files that directly import that module when no production module
imports it. Changes to test files select those files directly. Maintained
non-Python mappings may be declared in `.github/local-test-ownership.json`;
selectors there may be pytest files or nodes. This narrowing is deliberately
conservative: deletes, renames, untracked files, transitive production imports,
shared infrastructure, unknown paths, invalid manifests, and import parse
failures retain the owning suite commands. The focused selection is an edit-loop
aid; `make check` and CI remain the handoff gates.

| Change | Local handoff | CI adds |
| --- | --- | --- |
| Docs only | `make docs-linkcheck` | Documentation |
| Focused Python | affected target, then `make check` | Planned Python/static/package lanes |
| Benchmark task or verifier | `make harbor-check-task DATASET=... TASKS=...` and `make harbor-oracle-task DATASET=... TASKS=...` | Exact task contract and Oracle |
| Benchmark README or validation regression | focused Harbor checks | Contract checks; no Oracle |
| Lean runtime | focused `make test-lean`, then `make check` | Lean plus affected lanes |
| CI, dependencies, or unknown paths | `make check-static` plus affected tests | Fail-closed functional lanes |

Before final validation, use `make test-plan BASE=<revision>` to preview the
changed-path selection and run the selected checks on the final tree. If the
tree changes during validation, rerun checks whose evidence was invalidated by
that change; do not describe results from an earlier tree as final-tree
validation.

Parallel agents sharing one checkout must divide path ownership before editing.
They must not switch branches, stage, commit, clean, or rewrite shared files
while another agent is working. Integrate their edits first, then run the
planned checks on the final tree. Use isolated worktrees only when the workflow
explicitly assigns them.

Keep the local edit loop on directory-owned Make targets rather than inventing
marker filters:

```sh
make test-unit
make test-component TESTS=tests/component/capabilities/test_atomic_capabilities.py
make test-domain TESTS=tests/domain/graph/test_graph_invariant_domain.py
```

Ownership is by test directory. Lean tests live in the serial `lean` boundary
lane; keep them out of the normal xdist pools because Mathlib processes can
retain several gigabytes. CI installs the pinned Lean toolchain and Mathlib
cache in a dedicated runner.
Use `uv run --locked pytest --lf` after a failure, `uv run --locked pytest -n 0`
while debugging, and `make check` before handoff. Use
`make test-lean TESTS=tests/boundary/providers/lean/test_lean_repl_runtime.py` for
a deliberately focused local Lean reproduction, or dispatch the remote Lean
debug workflow from GitHub Actions when local Lean is impractical. Use
`make test-lean PYTEST_ARGS=--lf` to rerun a failed Lean-runtime test.
Do not use unfiltered `uv run pytest` as the normal complete-suite command
because it mixes Lean into the general xdist pool; pytest rejects that unsafe
combination with the corresponding `make` targets in its error message.
## Verification rules

- Do not turn a timeout, cancellation, error, incomplete enumeration, or
  missing witness into a mathematical conclusion.
- Do not promote an evaluator score, solver status, model answer, or search
  result directly to `VERIFIED`.
- Keep execution status, input validity, mathematical conclusion, assurance,
  and evidence type separate.
- Bind verified evidence to the exact claim, semantics, candidate, scope,
  certificate format, and checker identity.
- Keep checker authorization and trust policy outside untrusted plugins and
  search workers.

For trust-sensitive changes, write the failing invariant or attack test first
and verify replay through an independent checker process.

## Documentation

Documentation follows the [Diátaxis framework](https://diataxis.fr/). Place
documentation according to the reader's task:

- `docs/tutorials/` teaches through a complete guided experience;
- `docs/how-to/` explains how to complete one specific task;
- `docs/reference/` defines exact contracts and lookup information;
- `docs/explanation/` records architecture, rationale, and tradeoffs.

Domain-owned capability references live in `docs/reference/capabilities/<domain>/`.
Adding an operation or provider does not require editing a central documentation
list.

Keep rolling product goals separate from supported release behavior.
For hosted MCP changes, update and validate
[`docs/how-to/deploy-remote-mcp.md`](docs/how-to/deploy-remote-mcp.md) together
with any affected files under `deploy/`. Do not promote ignored `tmp/`
configuration or deployment notes into source-of-truth instructions.
For documentation-only changes, run:

```sh
git diff --check
git diff -- AGENTS.md README.md CONTRIBUTING.md docs/
make docs-linkcheck
```

Verify every relative Markdown link before submitting the change
(`make docs-linkcheck` checks project Markdown offline).

## Releases

The manifest-driven Release Please configuration keeps the Python and npm
package versions synchronized. CI tests and packs the npm launcher
independently, then publishes both distributions after a release is created.
The `jacobian` package on npm must authorize `.github/workflows/release.yml` as
its trusted GitHub Actions publisher, using the `npm` environment; releases use
OIDC rather than a long-lived npm token.

## Pull requests

Keep each change focused on one outcome. Explain the problem, the resulting
behavior or contract, any compatibility impact, and the validation performed.
Link a relevant issue when one exists. Include screenshots only when rendered
layout or diagrams materially change.

Open a new issue when review, conformance testing, or real use identifies a
specific unresolved behavior. Each issue should describe the observable
mathematical or operational problem, distinguish verified facts from
hypotheses, name the affected public contract or conformance case, include a
minimal reproduction or failing test where practical, and state whether the
change can affect artifact identity, checker authority, evidence binding, or
experiment integrity. Do not prescribe a solver or backend unless the
requirement depends on it. Do not open umbrella issues that restate product
goals; product goals become issues only when the problem and success criteria
are concrete.

## Test ownership and selection

Test directories define semantic ownership: `tests/unit`, `tests/component`,
`tests/domain`, `tests/composition`, `tests/boundary`, and `tests/e2e`. Use the
matching `make test-*` target as the canonical entry point. Markers are retained
only when they alter execution: `requires_provider(name)`, `performance`,
`property`, and `destructive_process`. They do not replace directory ownership.
Reproduce the scheduled validation lanes locally with `make test-stress` and
`make test-ordering ORDERING_LANE=domain PYTEST_ARGS=--randomly-seed=17`
(`ORDERING_LANE` selects the semantic lane to reseed; `domain` and
`composition` are the sharded lanes whose scheduled seed matters most).
Locked `pytest-repeat` and `pytest-randomly` are part of the dev environment.

Tests may reuse concept-specific helpers under `tests/support`, but must not
import helpers from a sibling semantic lane. Keep fixtures in the narrowest
directory or module that needs them, and keep support modules to ordinary data
builders or one stable test concept rather than hidden setup.

CI change impact is declared in `.github/ci-impact.json`. Its matching rules are
additive, so a path may require several suites. Integration timing history is a
scheduling hint produced by successful `main` runs; it is not committed state,
and missing or invalid history falls back to equal-weight sharding.
