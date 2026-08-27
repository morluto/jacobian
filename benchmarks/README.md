# Jacobian Harbor datasets

Every executable benchmark case is a self-contained Harbor task. Dataset
identity is an evaluation contract: it records what a result may support, not
which mathematical subject the task belongs to. Subject organization lives in
the required `primary_domain` taxonomy and the more detailed `field` metadata;
tasks remain direct children of their dataset roots.

See [benchmark documentation](docs/index.md) for task authoring, evaluation
methods, and operator-run observations.

`benchmarks/datasets/<dataset>/` is the Harbor dataset root and contains the
dataset's executable task bundles directly. `members/` retains Jacobian's
authoritative identity, provenance, assurance, provider, environment-profile,
verifier-contract, and evaluation-ownership metadata. `suite.toml` contains
stable dataset policy and defaults only.
Reusable Harbor infrastructure belongs under `benchmarks/tooling/`, adapters
under `benchmarks/adapters/`, and task-owned discovery context under each
task's `analysis/` directory. Immutable evaluation handoffs belong with the
snapshot or task that owns them; there is no active suite-wide research
evaluation bundle.

| Dataset | Purpose | Default execution |
| --- | --- | --- |
| `jacobian/mathematical-benchmarks-v1` | Fixed hidden-runtime mathematical tasks | Oracle and optional observation |
| `jacobian/symbolic-coordination-v1` | Exact polynomial-map coordination pilot | Oracle |
| `jacobian/public-reproductions-v1` | Replay known public mathematical cases | Oracle |
| `jacobian/conjecture-probes-v1` | Independently checked bounded conjecture progress | Oracle |
| `jacobian/examples-v1` | Tutorial and smoke workflows | Oracle |

`registry.toml` is the discovery index. Each dataset's member fragments own
membership. Intentional evaluation and publication events create immutable,
content-addressed locks under `benchmarks/snapshots/`; those locks bind the
suite header, ordered Harbor task digests, Harbor version, resolved images and
source tree, split, and evaluation configuration. Inventory diagnostics record
the digest of each task-local verifier support file separately. Harbor
publication `dataset.toml` files are generated under ignored `dist/harbor/`
from a lock and are never committed in dataset roots. Harbor jobs point at the
dataset root and use Harbor's native task-name filtering.

The repository `.uv-version` pins active development, CI, release, and product
image builds. Harbor task images remain bound to the uv version and digest in
their published task identity; changing that environment requires a new task
digest and Oracle validation.

Tasks expose only `instruction.md` and `environment/` to an evaluated agent.
Oracle solutions remain under `solution/`; verifier code and fixtures remain
under `tests/`. No compatibility directories or aliases for the former
benchmark layout are retained.

Each task owns its `tests/verifier_support.py` file because Harbor builds the
separate verifier image from that task's `tests/` directory. The task template
contains the current generic helper for newly scaffolded tasks; existing tasks
are not silently rewritten. `make harbor-sync DATASET=<id> TASKS="<ids>"`
updates only the selected verifier checksum labels. The checksum hashes
filenames, NUL separators, and bytes. The read-only Harbor gates
validate local support files and never synchronize unrelated tasks.

`benchmarks/tooling/public_contract.py` is internal repository tooling, not an
adapter. For `mathematical-benchmarks-v1`, each verifier owns one
`tests/public_contract.json`; the tool projects only the standard agent-visible
`instruction.md` and `environment/submission_schema.json` files. The checked-in
task-local declaration is copied into the separate verifier image so protocol
validation does not depend on files from the agent context.

## Commands

```sh
make harbor-plan BASE=origin/main
make harbor-execution-check
make harbor-check-task DATASET=mathematical-benchmarks-v1 TASKS="task-id"
make benchmark-inventory OUTPUT=/tmp/benchmark-inventory.json
make benchmark-snapshot DATASET=mathematical-benchmarks-v1
make benchmark-snapshot-validate LOCK=benchmarks/snapshots/mathematical-benchmarks-v1/<digest>.lock.json
make benchmark-publish LOCK=benchmarks/snapshots/mathematical-benchmarks-v1/<digest>.lock.json
make harbor-oracle-task DATASET=mathematical-benchmarks-v1 TASKS="task-id"
make harbor-check
make harbor-check-all  # explicit full host-verifier reproduction
make harbor-oracle DATASET=mathematical-benchmarks-v1 FULL=1
make harbor-oracle-all
uv run pytest tests/process/providers
make codex-visibility
```

`codex-visibility` is a separate opt-in agent-adoption diagnostic, not a Harbor
mathematical correctness task. It compares no Jacobian with Jacobian MCP only
and measures whether Codex discovers, inspects, invokes, completes, or
independently checks relevant operations. See
[Run the MCP visibility evaluation](docs/run-codex-visibility-evaluation.md).

`symbolic-coordination-v1` keeps its deterministic 26-case pilot separate
from the fixed `mathematical-benchmarks-v1` snapshots. Its task bundles are solvable
without Jacobian and do not yet include the later comparison harness.

`harbor-execution-check` is the focused local gate for job JSON, MCP
configuration, job-level Compose overlays, and their execution helpers. It
checks Harbor contracts and the owning tooling tests without running the
task-specific verifier regression corpus. Task
`environment/docker-compose.yaml` files are executable benchmark input, not
job overlays, and remain gated by `harbor-check-task` and
`harbor-oracle-task`. `harbor-check-task` and `harbor-oracle-task`
require an explicit dataset and task selection and are the normal gates for a
leaf task. The full `harbor-check`/`harbor-oracle` paths remain for shared
tooling, schemas, registry, suite policy, and control-plane changes.
`harbor-oracle` requires `TASKS` unless `FULL=1` is explicitly supplied;
`harbor-oracle-all` is the intentional full-portfolio sweep.

Pull requests run contract checks and exact Oracles for changed executable
tasks; large multi-task edits defer that matrix to the merge queue. Merge-queue
groups add affected-dataset or shared-infrastructure Oracle
coverage, while pushes to `main` repeat the deterministic contract gate without
duplicating those Docker jobs. The weekly and manually dispatched benchmark
workflow performs the full portfolio sweep; maintainers can request the same
scope on a pull request with `ci:benchmark-full`.

Changed tasks remain one Oracle job each. Affected-dataset and full-portfolio
sweeps use deterministic, dataset-bounded shards with at most four concurrent
jobs. Every shard carries the exact task IDs and Harbor digests it owns, and
the result validator still requires each selected task exactly once. The
planner accepts an optional positive-seconds timing file and otherwise falls
back to equal weights. Successful full runs on `main` publish median per-task
timings as an artifact and cache; later plans restore that uncommitted history
automatically. Timing hints older than 30 days (or dated in the future) are
rejected, so the planner safely returns to equal shard weights until a fresh
successful `main` run publishes replacement evidence. The merge job also
uploads a predicted-versus-actual report for every host shard.

Observation results are normalized into content-bound JSON before comparison.
Correctness, evidence validity, scope, assurance calibration, false
certification, tool traces, tokens, time, and cost remain separate. Reports
from the public workflow suite are workflow evidence only, never causal
operation evidence.

The committed three-attempt control and treatment jobs are manual
reproducibility fixtures. An operator may run `make agent-eval` with
`EVAL_EXECUTE=1`, then validate and compare the resulting evidence, but model
execution is not part of routine task authoring or the pull-request gate.

Private held-out evaluation is dispatched through the protected
`Held-out Benchmarks` workflow. New runs use manifest version 3. The immutable
Harbor snapshot lock is the authority for selected task membership and task
digests; the manifest binds that lock's ID, URI, and digest together with the
held-out archive URI and digest. It also freezes the runtime image, Jacobian
catalog and policy, Harbor and agent/model configuration, prompt, seed,
budgets, stages, and condition. Archived task digests must agree with the
referenced snapshot.

The workflow downloads the bundle with OIDC, refuses unpinned or unsafe
content, and uploads only non-Oracle evidence. Plans, ledgers, normalized
results, and comparisons bind the manifest digest rather than copying editable
provenance fields into another authority. Historical snapshots and records are
immutable; a new evaluation boundary creates a new snapshot and manifest.
The control condition explicitly disables Jacobian and is forbidden from
declaring an image, sidecar, or MCP server; only the treatment binds the
digest-pinned Jacobian image and advertised server, catalog, and policy
identities. Each task/repetition becomes a randomized C1/C2 pair of one-attempt
Harbor jobs. A resumable ledger binds the exact plan and checks token and cost
accounting after each complete pair. Because Harbor cannot currently hard-stop
Codex at those limits, missing accounting or a pair-boundary overage makes the
run incomplete and prevents a valid comparison.

Run `make heldout-smoke` to build a temporary non-mathematical private-bundle
fixture and exercise its rendered contract through Harbor's zero-model-cost
`nop` and `oracle` agents.

Performance timing is reported separately from reward, and research datasets
are explicitly non-comparative diagnostics. Uniform task structure does not
make rewards across these datasets comparable.

See [authoring a Harbor benchmark task](docs/author-harbor-benchmark-task.md),
[benchmark contracts](docs/benchmark-contracts.md), and the
[Harbor benchmarks skill](../.agents/skills/harbor-benchmarks/SKILL.md).

## Research loop and subject taxonomy

Broad mathematical benchmarks expose workflow gaps. Targeted contract suites
isolate one failure mode or mathematical obligation. Protected held-out runs
then test causal improvements on a frozen, non-public contract. Conjecture
probes occupy a separate rung: they score only independently checked bounded
progress on an explicit finite scope and never imply a global theorem.

Each member records a fixed `primary_domain` such as `graph-theory` alongside
its detailed `field` such as `graph-domination`. These labels support filtering
and inventory reporting; they are not scoring signals and do not prescribe a
tool, decomposition, or research strategy. The inventory tool accepts
`--dataset`, `--primary-domain`, and `--field`; unknown or empty selections fail
closed.

The active `agent-workflow-v1` identity was renamed to
`mathematical-benchmarks-v1`; no active compatibility alias exists. Its
historical `agent-workflow-v1` snapshot and ignored result directories remain
byte-for-byte evidence of the earlier evaluation boundary and are not rewritten.
