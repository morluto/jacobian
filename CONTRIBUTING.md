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

Most changes should begin with the CI-planned affected local validation:

```sh
make setup
make affected AFFECTED_BASE=origin/main
```

`make affected` uses the same checked-in planner as pull-request CI. Use
`make affected-plan` to inspect its selection without running it. Choose a
different command only when the change or current question fits one of these
cases:

| Situation | Command |
| --- | --- |
| One-owner edit loop with unrelated static drift | `make handoff-scoped LANE=... TESTS=... PATHS="..."` |
| A changed process, MCP, or Singular boundary | Add its named `make test-*` lane |
| Documentation | `make docs-linkcheck` |
| Frozen ordinary tree | `make check` once |
| Reproduce all ordinary CI or exhaustive evidence | `make check-all` or `make test-full` |

`make check` and `make check-all` take the worktree-local broad-validation
lease. Run `make validation-status` if one is already running. The
[testing strategy](docs/reference/testing-strategy.md) owns lane selection,
CI behavior, timing artifacts, and escalation details.

## Development environment

Jacobian uses Python 3.12 and the uv release pinned in [`.uv-version`](.uv-version).

```sh
make setup          # locked dev environment and Python backends
```

For command syntax, lanes, focused debugging, and the exceptional full-suite
path, use the [testing strategy](docs/reference/testing-strategy.md). `make
help` lists common commands; `make help-all` includes diagnostic plumbing.

Run `make hooks` once to install commit-time formatting, syntax, secret,
large-file, dead-code, and actionlint hooks plus the static
`make lint typecheck` pre-push gate. `make fix` applies Ruff's safe lint fixes
followed by formatting; `make precommit` then runs the broad ordinary gate.
Use `make handoff LANE=... TESTS=...` for the normal focused path instead.
Hooks remain bypassable for exceptional cases with Git's standard `--no-verify`
option.

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

Before final validation, run `make handoff LANE=... TESTS=...` for the changed
behavior. If the tree changes during validation, rerun checks whose evidence was
invalidated by that change; do not describe results from an earlier tree as
final-tree validation. `make check-all` is an explicit broad reproduction, not
a routine closeout requirement. Merge-group CI owns the complete ordinary matrix
and scale evidence.
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
only when they alter execution: `requires_backend(name)`, `property`,
`exhaustive`, and `scale`. They do not replace directory ownership. Scheduled
validation owns property, exhaustive, and scale evidence; keep a representative
behavioral case in the ordinary owning lane.

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
