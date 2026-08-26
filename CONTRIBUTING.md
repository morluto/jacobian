# Contributing to Jacobian

Jacobian is a pre-stable **math toolbox for agents**: atomic tools behind
`math.find` / `math.run`, math-first results, and agent-owned composition.
Contributions should preserve that product model—see
[product-blueprint](docs/explanation/product-blueprint.md) and
[architecture](docs/explanation/architecture.md).

## Before changing code

Read the [documentation home](docs/index.md), the
[product model](docs/explanation/product-blueprint.md), and the
[testing strategy](docs/reference/testing-strategy.md).
Use the installed catalog and current references for present tool membership.
Before proposing a public operation, follow the
[public operation admission](docs/reference/public-operation-admission.md)
contract and the
[operation contract review](docs/reference/domain-operation-library.md#operation-contract-review).

## Contributor quick path

Most changes need only the locked environment and the bounded local handoff:

```sh
make setup
make check
```

Then open a pull request. `make setup` installs the locked development
environment with the complete maintained Python backend stack. `make check`
runs Ruff, mypy, and the math, catalog, dispatch, CLI, and tooling
owners. Add the named `make test-*` lane for the behavior or boundary changed;
pull-request CI runs static checks plus the changed mathematical owner. Public
operation, model, admission, and canonical-contract changes also run catalog
conformance and every advertised invocation example; merge-group candidates
and `main` run the full mathematical suite.
Open the PR once it is green, and add any explicitly relevant specialist
validation called out below.

Route validation by the boundary changed:

| Change | Required local handoff |
| --- | --- |
| Ordinary operation or model | `make check` plus the owning domain tests |
| Public contract or catalog | Above plus the catalog conformance test |
| Singular adapter or codec | Above plus `make test-singular` |
| Shared process supervisor | Above plus `make test-process` |
| Documentation | `make docs-linkcheck` |

Singular is intentionally not part of `make check`: its pinned hosted-CI lane
provides the required runtime evidence without making the executable a
prerequisite for every developer loop.

`make quick` is the cheaper loop: it omits mypy but runs the same
owner tests. The
pre-push hook stays `make lint typecheck`. Focused debugging uses
`uv run pytest path/to/test.py`. Default `uv run pytest` collects the ordinary
testpaths`; it does not run process or MCP trees.

CI runs static checks, the selected ordinary Python surface, MCP boundaries,
and the wheel smoke. Full mathematical suites run on merge-group candidates and
on `main`, not on every pull request.

Specialist lanes (`make test-process` and `make test-mcp`) are troubleshooting
and boundary work, not a routine confidence gate. Run one only when your change
crosses that boundary or you are reproducing an environment-specific failure. The
[testing strategy](docs/reference/testing-strategy.md) is the authoritative
source for the change matrix, directory ownership, and the escalation rules.

Coverage follows the same ownership rule. Ordinary lanes collect parent-process
branch coverage; do not add child-worker coverage plumbing to an inline
operation.

### When the quick path is not enough

- **Documentation only:** `make docs-linkcheck` is the dedicated lane; CI runs
  it too. See [Documentation](#documentation). Static and boundary CI evidence
  still runs, but no mathematical owner is selected.
- **Broad or unknown impact** (CI, dependencies, shared infrastructure): run
  `make check-static` plus the affected tests, and let CI own the fail-closed
  functional lanes.
- **Maintained Python libraries:** run the owning `tests/math` test when a
  direct mathematical backend changes.
- **Exhaustive local reproduction:** `make test-full` is an explicit exception
  path, not a routine gate. It takes this worktree's exhaustive validation
  lock; `make validation-status` shows whether that lock is held. Before it,
  verify that no other pytest or delegated-agent validation is running on the
  host, and never assign it to a parallel agent sharing the checkout. The
manually dispatched Python Debug workflow reproduces one
  pytest file or node in a prepared remote environment when the relevant local
  runtime is impractical.

## Development environment

Jacobian uses Python 3.12 and the uv release pinned in [`.uv-version`](.uv-version).

```sh
make setup          # locked dev environment and Python backends
```

`make check` is the bounded lint, type, and owner handoff;
`make quick` omits typechecking for a shorter edit loop. For an ordinary edit
loop, name both the semantic owner and changed test path:

```sh
make test-focused LANE=math TESTS=tests/math/graphs/test_graph_distance_matrix.py
```

`LANE` is one of `math`, `catalog`, `dispatch`, `cli`, `tooling`,
`integration`, `process`, or `mcp`. The command keeps each lane's established
timeout and worker count; it does not select a lane from a path implicitly.
`make check-all` explicitly reproduces every ordinary Python CI lane. The
pre-push hook intentionally runs only
`make lint typecheck` so it stays below the interactive feedback budget.
`make check-static` adds dependency/dead-code checks and a package build when a
focused change needs them. Run `make help` for the common command index and
`make help-all` for lifecycle and diagnostic plumbing.

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
options through `PYTEST_ARGS`; `test-focused` is the discoverable form for an
explicit owner plus path. They print their ten slowest tests by default
(override with `PYTEST_DIAGNOSTIC_ARGS=--durations=0`). Use
`uv run --locked pytest --lf` after a failure and `uv run --locked pytest -n 0`
while debugging. Default `uv run pytest` omits process and MCP trees; use
`make test-process` or `make test-mcp` for those boundaries.
See the [testing strategy](docs/reference/testing-strategy.md) for the canonical
lane commands and narrowing examples.

### Parallel agents sharing a checkout

Parallel agents sharing one checkout must divide path ownership before editing.
They must not switch branches, stage, commit, clean, or rewrite shared files
while another agent is working. Integrate their edits first, then run the
planned checks on the final tree. Use isolated worktrees only when the workflow
explicitly assigns them.

For parallel pull-request work, each writer owns one isolated worktree and
one branch at a time. Record an issue or PR claim before implementation; check
for a current claim and for sibling public-operation IDs before beginning the
same scope. Fetch immediately before pushing and inspect a changed head rather
than retrying equivalent work. Do not recreate a merged or closed pull-request
branch: start a follow-up branch instead. The agent-facing version of this
protocol, including conflict-resolution checks, lives in `AGENTS.md`.

Before final validation, run `make check` plus the named lane that owns the
changed behavior. If the tree changes during validation, rerun checks whose
evidence was invalidated by that change; do not describe results from an
earlier tree as final-tree validation. `make check-all` is an explicit broad
reproduction, not a routine closeout requirement. Merge-group CI owns the
complete mathematical matrix.
Run the owning mathematical test when a maintained Python backend changes.

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

`make harbor-check` validates repository-wide Harbor contracts (job JSON, MCP
config, job-level Compose overlays, adapters, and execution helpers) and the
unit tests that own them; it deliberately excludes unrelated task-specific
verifier regressions. `make harbor-check-all` is the explicit full integration
reproduction and takes the same worktree validation lock as other exhaustive
local targets. `make harbor-plan` writes one canonical `plan.json` from the
normalized changed-path list; temps live only inside
the recipe. Task `environment/docker-compose.yaml` files are
executable benchmark input, not job overlays, and remain gated by
`make harbor-check-task` and `make harbor-oracle-task`. Use
`make harbor-plan BASE=origin/main` for benchmark contracts and Oracle scope;
run it through Make because the planner requires the pinned Harbor runtime to
compute task digests.

Current GitHub Actions identity is the workflow YAML on the default branch.
Historical registrations whose files are gone, including leftover
`agent-port-*` and `agent-rebase-*` workflows, stay disabled in the GitHub UI
with their run history retained; do not add an auto-disable bot.
`python tools/inventory_github_workflows.py` is the non-mutating inventory.
Branch protection should require the CI check named `required`.

Benchmark and evaluation material is not part of the Jacobian product
documentation. Keep any such work isolated from the server's operation
contracts and validate it through its own repository-local workflow.

## Bounded-result rules

- Do not turn a timeout, cancellation, error, incomplete enumeration, or
  missing witness into a mathematical conclusion.
- Keep execution status, input validity, and the domain mathematical conclusion
  separate.
- Do not promote an evaluator score, solver status, model answer, or search
  result beyond the conclusion stated by its typed domain result.

For trust-sensitive changes, write the failing invariant or attack test first.

## Documentation

Documentation follows the [Diátaxis framework](https://diataxis.fr/). Place
documentation according to the reader's task:

- `docs/how-to/` explains how to complete one specific task;
- `docs/reference/` defines exact contracts and lookup information;
- `docs/explanation/` records architecture, rationale, and tradeoffs.

The installed catalog is the operation reference. Add prose only when an
external boundary needs context that a generated schema cannot express.

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
hypotheses, name the affected public contract or conformance case, and include
a minimal reproduction or failing test where practical. Do not prescribe a
solver or backend unless the requirement depends on it. Do not open umbrella
issues that only restate the product model; open issues when the problem and
success criteria are concrete.

## Test ownership and selection

Test directories mirror their semantic owners: `tests/math`, `tests/catalog`,
`tests/dispatch`, `tests/cli`, `tests/tooling`, and `tests/integration`, with
separate `tests/process` and `tests/mcp` boundary owners. Use the matching
`make test-*` target as the canonical entry point. Markers are retained
only when they alter execution: `requires_backend(name)`, `property`, and
`exhaustive`. They do not replace
directory ownership. Scheduled validation owns `make test-exhaustive`; keep a
representative behavioral case in the ordinary owning lane.

Lane execution follows those owners. MCP and process stay on named Make targets
because they exercise transport and kill-safe process boundaries.
Prefer a direct domain test, then a focused MCP test only when the public
projection changes.

Tests may reuse concept-specific helpers under `tests/support`, but must not
import helpers from a sibling semantic lane. Keep fixtures in the narrowest
directory or module that needs them, and keep support modules to ordinary data
builders or one stable test concept rather than hidden setup.

The [testing strategy](docs/reference/testing-strategy.md) is the authoritative
source for the change matrix, the canonical lane commands, directory ownership,
and the specialist-lane escalation rules.
