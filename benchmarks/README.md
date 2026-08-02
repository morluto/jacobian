# Jacobian Harbor datasets

Every executable benchmark case is a self-contained Harbor task. The six
dataset identities below keep workflow observations, public reproductions,
research diagnostics, operational measurements, provider feasibility, and
examples from making incompatible claims look comparable.

`benchmarks/datasets/<dataset>/` is the Harbor dataset root and contains the
dataset's executable task bundles directly. `members/` retains Jacobian's
authoritative identity, provenance, assurance, provider, environment-profile,
verifier-contract, and evaluation-ownership metadata. `suite.toml` contains
stable dataset policy and defaults only.
Reusable Harbor infrastructure belongs under `benchmarks/tooling/`, adapters
under `benchmarks/adapters/`, and non-runnable evaluation plans and research
handoffs under `research/evaluations/`.

| Dataset | Purpose | Default execution |
| --- | --- | --- |
| `jacobian/agent-workflow-v1` | Fixed Jacobian-enabled mathematical workflows | Oracle and optional observation |
| `jacobian/public-reproductions-v1` | Replay known public mathematical cases | Oracle |
| `jacobian/research-diagnostics-v1` | Answer-visible research challenges | Oracle diagnostics |
| `jacobian/performance-v1` | Historical pinned runtime baseline | Oracle |
| `jacobian/provider-feasibility-v1` | Pinned optional-backend checks | Oracle |
| `jacobian/examples-v1` | Tutorial and smoke workflows | Oracle |

`registry.toml` is the discovery index. Each dataset's member fragments own
membership. Intentional evaluation and publication events create immutable,
content-addressed locks under `benchmarks/snapshots/`; those locks bind the
suite header, ordered Harbor task digests, Harbor version, resolved images and
verifier runtime, source tree, split, and evaluation configuration. Harbor
publication `dataset.toml` files are generated under ignored `dist/harbor/`
from a lock and are never committed in dataset roots. Harbor jobs point at the
dataset root and use Harbor's native task-name filtering.

The repository `.uv-version` pins active development, CI, release, and product
image builds. Harbor task images remain bound to the uv version and digest in
their published task identity; changing that environment requires a new task
digest and Oracle validation. In particular, `performance-v1` declares its
historical source and toolchain in `baseline.toml` rather than pretending to
measure current main.

Tasks expose only `instruction.md` and `environment/` to an evaluated agent.
Oracle solutions remain under `solution/`; verifier code and fixtures remain
under `tests/`. No compatibility directories or aliases for the former
benchmark layout are retained.

## Commands

```sh
make harbor-plan BASE=origin/main
make harbor-check
make benchmark-inventory OUTPUT=/tmp/benchmark-inventory.json
make benchmark-snapshot DATASET=agent-workflow-v1
make benchmark-snapshot-validate LOCK=benchmarks/snapshots/agent-workflow-v1/<digest>.lock.json
make benchmark-publish LOCK=benchmarks/snapshots/agent-workflow-v1/<digest>.lock.json
make harbor-oracle DATASET=agent-workflow-v1 TASKS="task-id"
make harbor-oracle-all
make agent-eval DATASET=agent-workflow-v1 TASKS=graph-counterexample EVAL_EXECUTE=1
make agent-eval-validate RESULTS=... JOB=... RUNTIME_SNAPSHOT=... CONDITION=control OUTPUT=control-evidence.json
make agent-eval-compare CONTROL=control-evidence.json TREATMENT=treatment-evidence.json OUTPUT=report
make performance-eval
make provider-eval PROVIDER=cgal
```

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
automatically.

Observation results are normalized into content-bound JSON before comparison.
Correctness, evidence validity, scope, assurance calibration, false
certification, tool traces, tokens, time, and cost remain separate. Reports
from the public workflow suite are workflow evidence only, never causal
capability evidence.

Private held-out evaluation is dispatched through the protected
`Held-out Benchmarks` workflow. Its S3 manifest freezes the treatment image,
catalog and policy digests, task and Oracle identities, model, prompt, budget,
randomization, and pilot/decision sample sizes. The workflow downloads it with
OIDC, refuses unpinned or unsafe bundles, and uploads only non-Oracle evidence.
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

See [authoring a Harbor benchmark task](../docs/how-to/author-harbor-benchmark-task.md),
[reference benchmarks](../docs/reference/benchmarks.md), and the
[Harbor benchmarks skill](../.agents/skills/harbor-benchmarks/SKILL.md).
