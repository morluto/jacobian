# Contributing to Jacobian

Jacobian is a pre-stable **math toolbox for agents**: atomic tools behind
`math.find` / `math.run`, math-first results, and agent-owned composition.
Contributions should preserve that product model—see
[product-blueprint](docs/explanation/product-blueprint.md) and
[architecture](docs/explanation/architecture.md).

## Find the relevant guidance

Use [docs/index.md](docs/index.md) to find task-specific documentation. Read the
[product model](docs/explanation/product-blueprint.md) and
[architecture](docs/explanation/architecture.md) when changing product scope or
ownership. For public-operation proposals, follow
[public operation admission](docs/reference/public-operation-admission.md);
for operation contracts or implementations, use the applicable
[operation review](docs/reference/domain-operation-library.md#operation-contract-review)
sections. The installed catalog owns current operation membership.

## Contributor quick path

For mathematical changes, review three things before merging:

1. **Independent correctness evidence:** state the defining identity, its
   conventions, and a test that breaks a plausible wrong implementation.
2. **The complete execution path:** identify who establishes each invariant,
   where candidates become trusted results, and what work or expansion repeats.
3. **A useful accepted boundary:** demonstrate a representative valid request
   as well as an excessive request that is rejected. Rejection tests alone do
   not establish that an operation remains usable.

For an admission or scale defect, scale first: improve the estimate,
representation, reduction, algorithm, or backend so the motivating valid
request succeeds. Do not turn a cheaply executable request into a permanent
rejection regression. Follow the
[execution-envelope review](docs/reference/public-operation-admission.md#execution-envelope-review)
before retaining a limit.

Record the applicable evidence in the existing PR description; do not add a
new review artifact or framework. The
[operation review](docs/reference/domain-operation-library.md#operation-contract-review),
[test evidence guidance](docs/reference/testing-strategy.md#evidence-plans-for-exact-operations),
and [backend contract](docs/reference/mathematical-backends.md#common-adapter-obligations)
define these checks.

For code changes, run CI-planned affected validation on the final tree:

```sh
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
| Broad ordinary validation requested or needed for cross-cutting evidence | `make check` once on the frozen tree |
| Add ordinary integration or reproduce all local semantic test lanes | `make check-all` or `make test-full` |

`make check` and `make check-all` take the worktree-local broad-validation
lease. Run `make validation-status` if one is already running. The
[testing strategy](docs/reference/testing-strategy.md) owns lane selection,
CI behavior, timing artifacts, and escalation details.

## Development environment

Jacobian uses Python 3.12 and the uv release pinned in [`.uv-version`](.uv-version).

```sh
make setup          # locked dev environment and Python backends
```

Install the separate Singular and QEPCAD system executables using the
[backend requirements](docs/how-to/backend-requirements.md). `make setup` does
not install operating-system packages; the service image includes both.

For command syntax, lanes, focused debugging, and the exceptional full-suite
path, use the [testing strategy](docs/reference/testing-strategy.md). `make
help` lists common commands; `make help-all` includes diagnostic plumbing.

Run `make hooks` once to install commit-time formatting, syntax, secret,
large-file, dead-code, and actionlint hooks plus the static
`make lint typecheck` pre-push gate. `make fix` applies Ruff's safe lint fixes
followed by formatting; `make precommit` then runs the broad ordinary gate.
Use `make affected` for branch validation or `make handoff LANE=... TESTS=...`
for a focused owner check.
Hooks remain bypassable for exceptional cases with Git's standard `--no-verify`
option.

On macOS, read the
[Z3 installation guide](docs/how-to/troubleshoot-z3-macos.md) before
troubleshooting a source-build failure from `uv sync --dev`.

For focused test syntax and specialist lanes, use the
[testing strategy](docs/reference/testing-strategy.md). Default `uv run pytest`
omits process and MCP trees; those boundaries have named Make targets.

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
branch: start a follow-up branch instead. See `AGENTS.md` for public-operation
collision and catalog-conflict checks.

Run focused owner checks while editing, then planner-selected validation on
the final tree. If the tree changes during validation, rerun checks whose
evidence was invalidated by that change; do not describe results from an earlier tree as
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
tests serially, then runs each selected Oracle serially and writes run evidence.
Preparation starts neither Oracle nor a model; validation starts Oracle, but no
model agent.

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
git diff -- AGENTS.md README.md CONTRIBUTING.md docs/ .github/
make docs-linkcheck
```

Verify every relative Markdown link before submitting the change
(`make docs-linkcheck` checks the root guides and `docs/` offline). For changes
to `.github/` templates, also check their relative links and issue-template YAML
front matter; the docs target does not traverse that directory.

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

### Review and closeout

Treat a concrete adversarial review finding as contract evidence, not as a
request for a cosmetic patch. Reproduce the reported behavior, identify the
cheapest boundary that can reject or represent it correctly, and add the
smallest behavioral regression that would fail without the fix. Depending on
the finding, that proof may need to cover an accepted near-limit request,
malformed or deeply nested input, generated-schema/runtime parity, native/MCP
parity, a producer-consumer round trip, a worker projection, or a deadline and
serialization phase after backend execution.

Before replying that a thread is resolved, inspect the complete final diff
against the intended base, resolve merge conflicts, and freeze the behavioral
tree. Run the affected owner and boundary lanes, then rerun checks invalidated
by formatting, generated files, conflict resolution, or later commits. When
automatic linting changes a pull request, use the CI run for the exact final
head SHA; skipped, withheld, duplicate, or older event records are not evidence
for a different revision. The handoff should name the final revision, tests
actually run, optional-backend skips, and any remaining proof gap.

## Test ownership and selection

The [testing strategy](docs/reference/testing-strategy.md) owns the change
matrix, lane commands, fixture ownership, specialist escalation, and exact
mathematical evidence. Select its sections for the behavior being changed.
