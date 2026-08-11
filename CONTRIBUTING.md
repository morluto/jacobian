# Contributing to Jacobian

Jacobian is a pre-stable 0.6.0 **math toolbox for agents**: atomic tools behind
`math.find` / `math.run`, math-first results, agent-owned composition, and
optional checker tools as **separate catalog IDs** (not dual-mode producers).
Contributions should preserve that product model—see
[product-blueprint](docs/explanation/product-blueprint.md) and
[architecture](docs/explanation/architecture.md).

## Before changing code

Read the [documentation home](docs/index.md), the
[product model](docs/explanation/product-blueprint.md), and the
[testing strategy](docs/reference/testing-strategy.md).
Use the installed catalog and current references for present tool membership.

## Contributor quick path

Most changes need only a light environment and the changed-path gate:

```sh
make setup PROFILE=core
make check-changed BASE=origin/main
```

Then open a pull request. `make setup PROFILE=core` installs the locked
development environment and requires only the core provider surface (NetworkX,
SymPy, and Z3). `make check-changed BASE=origin/main`
runs format, type-checking, and the exact tests selected from your changed
paths against that base ref. Open the PR once it is green, and add any
explicitly relevant specialist validation called out below.

CI owns the expensive correctness surface so the local loop stays fast. The
hosted pipeline owns the supported Python and OS matrices, the full Lean and
optional-provider environments, coverage enforcement, the compatibility smoke
suite, packaging, the security audit, duplicate-code detection, and the
exhaustive semantic-lane matrix. You do not need to reproduce those locally for
a routine change.

Specialist lanes (`make test-lean`, `make test-provider`, `make test-storage`,
`make test-process`, `make test-mcp`, `make test-e2e`, `make test-domain`, and
`make test-composition`) are troubleshooting and boundary work, not a routine
confidence gate. Run one only when your change crosses that boundary or you are
reproducing an environment-specific failure. The
[testing strategy](docs/reference/testing-strategy.md) is the authoritative
source for the change matrix, lane ownership, planning entry points, CI
classification, and the escalation rules.

### When the quick path is not enough

- **Documentation only:** `make docs-linkcheck` is the dedicated lane; CI runs
  it too. See [Documentation](#documentation).
- **Broad or unknown impact** (CI, dependencies, shared infrastructure): run
  `make check-static` plus the affected tests, and let CI own the fail-closed
  functional lanes.
- **Exhaustive local reproduction:** `make test-all-ci` is an explicit exception
  path, not a routine gate. Before it, verify that no other pytest or
  delegated-agent validation is running on the host, and never assign it to a
  parallel agent sharing the checkout. The manually dispatched Python Debug and
  Lean Debug workflows reproduce one pytest file or node in a prepared remote
  environment when the relevant local runtime is impractical.

Preview the exact local test selectors with `make test-plan BASE=origin/main`
and the hosted CI lane decision with `make ci-plan BASE=origin/main`. The two
reports answer different questions: the local plan may run exact importing
tests, while the hosted plan owns required semantic lanes and fail-closed
infrastructure coverage and emits a provenance-bound plan receipt. The full
command inventory, narrowing examples, diagnostic-duration overrides, and CI
classification detail live in the
[testing strategy](docs/reference/testing-strategy.md).

## Development environment

Jacobian uses Python 3.12 and the uv release pinned in [`.uv-version`](.uv-version).

```sh
make setup PROFILE=core          # locked dev environment; core readiness
make setup PROFILE=full-python    # also require every maintained Python extra
```

`make check` runs Ruff, mypy, and the unit lane; it is a useful local handoff.
The pre-push hook intentionally runs only `make lint typecheck` so it stays
below the interactive feedback budget. `make check-changed` combines the static
edit-loop checks with exact changed-path test selection; `make check-static`
adds dependency/dead-code checks and a package build when a focused change needs
them. Run `make help` for the common command index and `make help-all` for
lifecycle and diagnostic plumbing.

Run `make hooks` once to install commit-time formatting, syntax, secret,
large-file, dead-code, and actionlint hooks plus the static
`make lint typecheck` pre-push gate. `make fix` applies Ruff's safe lint fixes
followed by formatting; `make precommit` applies those fixes and then runs the
routine handoff checks. Hooks remain bypassable for exceptional cases with Git's
standard `--no-verify` option.

On macOS, read the
[Z3 installation guide](docs/how-to/troubleshoot-z3-macos.md) before
troubleshooting a source-build failure from `uv sync --dev`.

Every `make test-*` target accepts `TESTS=<file-or-node>` and extra pytest
options through `PYTEST_ARGS`, and prints its ten slowest tests by default
(override with `PYTEST_DIAGNOSTIC_ARGS=--durations=0`). Use
`uv run --locked pytest --lf` after a failure and `uv run --locked pytest -n 0`
while debugging. Do not use unfiltered `uv run pytest` as the complete-suite
command: it mixes Lean into the general xdist pool, and pytest rejects that
unsafe combination with the corresponding `make` targets in its error message.
See the [testing strategy](docs/reference/testing-strategy.md) for the canonical
lane commands and narrowing examples.

### Parallel agents sharing a checkout

Parallel agents sharing one checkout must divide path ownership before editing.
They must not switch branches, stage, commit, clean, or rewrite shared files
while another agent is working. Integrate their edits first, then run the
planned checks on the final tree. Use isolated worktrees only when the workflow
explicitly assigns them.

Before final validation, use `make test-plan BASE=<revision>` to preview the
changed-path selection and run the selected checks on the final tree. If the
tree changes during validation, rerun checks whose evidence was invalidated by
that change; do not describe results from an earlier tree as final-tree
validation. `make check-changed` is the normal local handoff; CI owns the
exhaustive evidence described above.

## Harbor and Oracle validation

Benchmark validation is decomposed into evidence roles even though CI shares
one checkout for the deterministic contract gate. A task README is documentation;
a task instruction, environment, manifest, or member record is executable
evaluation input. Shared environment profiles and execution-control changes may
escalate to merge-queue portfolio evidence.

For task authoring, `make harbor-prepare-task DATASET=... TASKS="..."` is the
explicitly mutating preparation step: it formats only Python owned by the
selected task and its dedicated validation leaf, runs scoped public-contract
and verifier-checksum synchronization, and reports every generated file that
changed. Follow it with
`make harbor-validate-task DATASET=... TASKS="..."` for the complete
source-read-only leaf gate, which resolves membership and planner selectors
once, fails fast through static quality and contracts, runs the selected host
tests serially, then runs each exact Oracle serially. Neither command starts an
Oracle or model.

`make harbor-execution-check` validates repository-wide Harbor contracts (job
JSON, MCP config, job-level Compose overlays, execution helpers) and the unit
tests that own them; it deliberately excludes the task-specific verifier
regressions under `benchmarks/validation/`, where `make harbor-check` retains
the full integration role. Task `environment/docker-compose.yaml` files are
executable benchmark input, not job overlays, and remain gated by
`make harbor-check-task` and `make harbor-oracle-task`. Use
`make harbor-plan BASE=origin/main` for benchmark contracts and Oracle scope;
run it through Make because the planner requires the pinned Harbor runtime to
compute task digests.

For the exact task authoring workflow and verifier changes, use the
[`harbor-benchmarks`](.agents/skills/harbor-benchmarks/SKILL.md) skill. The
[testing strategy](docs/reference/testing-strategy.md) and
[benchmark contracts](docs/reference/evaluations/benchmark-contracts.md) define
the full ownership, host-validation sharding, and Oracle semantics.

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

Keep product intent (product model / architecture) separate from supported
release behavior.
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
requirement depends on it. Do not open umbrella issues that only restate the
product model; open issues when the problem and success criteria are concrete.

## Test ownership and selection

Test directories define semantic ownership: `tests/unit`, `tests/component`,
`tests/domain`, `tests/composition`, `tests/boundary`, and `tests/e2e`. Use the
matching `make test-*` target as the canonical entry point. Markers are retained
only when they alter execution: `requires_provider(name)`, `performance`,
`property`, and `destructive_process`. They do not replace directory ownership.

Lane execution and CI path-impact rules are authored in
[`tests/plan_manifest.toml`](tests/plan_manifest.toml) and compiled to
[`tests/topology.toml`](tests/topology.toml) and
[`.github/ci-impact.json`](.github/ci-impact.json) via `make compile-test-plan`.
Do not hand-edit the generated projections. Prefer the hydration ladder in the
[testing strategy](docs/reference/testing-strategy.md): domain services before
`attached_complete_runtime` before `authorized_complete_runtime` before
`fresh_complete_runtime`. Inventory complete-runtime usage with
`make test-runtime-inventory`.

Tests may reuse concept-specific helpers under `tests/support`, but must not
import helpers from a sibling semantic lane. Keep fixtures in the narrowest
directory or module that needs them, and keep support modules to ordinary data
builders or one stable test concept rather than hidden setup.

The [testing strategy](docs/reference/testing-strategy.md) is the authoritative
source for the change matrix, the canonical lane commands, planning entry
points, CI classification, shard scheduling, marker policy, and the
specialist-lane escalation rules. Compiled impact matching rules are additive,
so a path may require several suites. Integration timing history is a
scheduling hint produced by successful `main` runs; it is not committed state,
and missing or invalid history falls back to equal-weight sharding.
