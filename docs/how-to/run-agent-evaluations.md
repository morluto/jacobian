# Run agent evaluations

[Documentation home](../index.md) · [Evaluation reference](../reference/evaluations/evaluation-methods.md)

This guide covers explicit operator-run Harbor observations and paired Jacobian
control/treatment runs. Model execution is not a routine
development or pull-request gate. The evaluation roles, assurance rules, and
interpretation boundaries are in the [reference page](../reference/evaluations/evaluation-methods.md).

## Platform note: offline Oracle on macOS Docker Desktop

Authoritative Docker Oracle validation runs on **Linux** (GitHub Actions and
Linux Docker hosts). Harbor tasks use `network_mode = "no-network"` for offline
isolation. Docker Desktop on macOS may reject that policy when the LinuxKit VM
lacks the nftables features Harbor's egress control requires; the trial fails
during environment validation before any container starts and must not be
treated as a task pass or fail. Prefer a Linux runner for Oracle and selective
backfills. Do not weaken task `no-network` settings merely to green local macOS
execution.

## Validate the dataset

Preview the benchmark plan before spending Docker or model time:

```sh
make harbor-plan BASE=origin/main
```

Then run the exact selected-task contract checks. `make harbor-oracle-task`
depends on `make harbor-check-task`, so the contract check runs first and
fails fast; the separate `make harbor-check-task` below is an optional
cheaper preview that avoids Harbor runtime startup:

```sh
make harbor-check-task DATASET=mathematical-benchmarks-v1 TASKS="graph-counterexample"
make harbor-oracle-task DATASET=mathematical-benchmarks-v1 TASKS="graph-counterexample"
```

Use `make harbor-check` and explicitly scoped `make harbor-oracle` paths for
shared Harbor contracts, schemas, registry, suite policy, or other control-plane
changes. Use `make harbor-check-all` only for shared verifier-harness changes or
an intentional portfolio-wide reproduction. Pass `TASKS="..."` for a bounded dataset Oracle;
pass `FULL=1` only when a complete dataset sweep is intentional.

Oracle attempts are serialized on a shared Docker host and receive a unique
job name plus a content-bound preparation receipt. The receipt includes task
digests, Harbor and Docker runtime versions, the Oracle job digest, and
normalized execution arguments. On Docker Engine versions older than 23, the
Make target selects the classic builder because the bundled legacy BuildKit
can reuse a same-named Dockerfile from another task through Compose Bake. Set
`HARBOR_ORACLE_DOCKER_BUILD_MODE` only for an explicit compatibility test; the
resolved mode remains part of the recorded experiment identity.

For changes limited to Harbor job JSON, MCP configuration, job-level Compose
overlays, or their execution helpers, use the focused local handoff instead:

```sh
make harbor-execution-check
```

This checks repository Harbor contracts and the execution-configuration unit
tests without running the task-specific verifier regression corpus, an Oracle,
Docker, or a model. Task `environment/docker-compose.yaml` files are
executable benchmark input, not job overlays; they remain gated by
`make harbor-check-task` and `make harbor-oracle-task`.

Task README and `benchmarks/validation/` changes do not require an Oracle;
they affect documentation or deterministic host-side validation. Changes to a
task's executable input or clean-room verifier do require the exact selected
task Oracle after contract validation.

## Set shared run conditions

For a paired comparison, set the same model, authentication, task filter,
prompt, budget, and environment before running either condition. The default
run is direct networking. Use `JACOBIAN_EVAL_PROXY=1` only when the host or
region requires a proxy for Codex's outbound model connection.

If a run needs a host proxy for model access, set the flag before both
conditions. The flag automatically reuses standard
`HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` variables; use the explicit
`JACOBIAN_EVAL_*` variables when the evaluation should use different values.
The proxy profile chooses `JACOBIAN_EVAL_HTTPS_PROXY`, then
`JACOBIAN_EVAL_ALL_PROXY`, then `JACOBIAN_EVAL_HTTP_PROXY`, and accepts
`http://`, `socks5://`, or `socks5h://` upstream URLs.
When inherited proxy URLs use `localhost` or `127.0.0.1`, the Makefile maps
them to `host.docker.internal` for the container automatically:

```sh
export JACOBIAN_MODEL='your-model'
export CODEX_FORCE_AUTH_JSON=1

export JACOBIAN_EVAL_HTTP_PROXY='http://host.docker.internal:7890'
export JACOBIAN_EVAL_HTTPS_PROXY='http://host.docker.internal:7890'
export JACOBIAN_EVAL_PROXY=1
```

Unset `JACOBIAN_EVAL_PROXY` and the proxy variables for direct networking. On
Linux, the host proxy must accept connections from Docker's bridge interface;
a proxy listening only on `127.0.0.1` is not reachable from the container.

This proxy profile is optional host integration, not part of a task contract.
It renders a private runtime configuration with both Harbor's transparent
egress listener and a loopback HTTP proxy chained to the upstream proxy, so
Harbor's task allowlist stays in force and upstream DNS resolution works for
proxy-aware clients. Codex uses the loopback listener. Codex and Jacobian share
Harbor's controlled network namespace, and MCP uses its loopback interface, so
local tool traffic never enters the upstream chain.

Proxy jobs also mount the host's standalone Codex executable to avoid a
trial-time package install. When `codex` is installed through npm, the runner
automatically resolves the bundled native Linux executable instead of mounting
the JavaScript launcher without Node. Set `JACOBIAN_EVAL_CODEX_BINARY` only to
override that candidate; the resolved file must be an executable Linux ELF.
Keep machine-specific Clash addresses in local environment variables rather
than committed configuration.

## Run with Jacobian

The local path uses Harbor's Docker environment, a Jacobian Compose sidecar,
and Harbor's external MCP configuration. The treatment contract is deliberately
small: direct runs address Jacobian by its Compose service name, while the
controlled proxy profile uses loopback because both containers share Harbor's
egress namespace. Codex web search is disabled in both profiles.

Choose the image from the source tree state. A clean revision pulls its
published full-SHA tag and resolves it to an immutable OCI digest. A dirty tree
builds `jacobian:local`, reusing Docker's normal local layer cache:

```sh
export JACOBIAN_IMAGE="$(make --no-print-directory eval-image)"
```

Published images live at `ghcr.io/morluto/jacobian`. The `main` and version-tag
workflow publishes both a human-readable `sha-<full-commit>` tag and registry
attestations, but evaluations use the returned `name@sha256:...` reference.
Pull-request builds exercise the Dockerfile without publishing an image. OCI or
Docker image archives are never repository artifacts and must not be committed.

Create the normal runtime snapshot before execution, then pass its path to the
run. `agent-eval` inspects the selected image immediately before Harbor starts
and adds its source SHA, OCI digest (or local image ID), platform, dirty flag,
and Jacobian package version. A Jacobian-enabled observation made from a dirty
or non-digest-pinned image remains useful for development, but normalization
fails closed rather than calling it reproducible evidence.

```sh
export SNAPSHOT_LOCK='benchmarks/snapshots/mathematical-benchmarks-v1/26e558abcfda80f944ff1659f73b3c89b22ed4ddd2700d8340c067dc4ed7b323.lock.json'
export RUNTIME_SNAPSHOT='benchmarks/results/my-run/runtime.json'
mkdir -p "$(dirname "$RUNTIME_SNAPSHOT")"
jq --arg model "$JACOBIAN_MODEL" \
  '{snapshot_id, harbor_version: "0.20.0", model: $model,
    condition: {id: "treatment", role: "PRIMARY_TREATMENT",
                jacobian_enabled: true}}' \
  "$SNAPSHOT_LOCK" > "$RUNTIME_SNAPSHOT"

make agent-eval DATASET=mathematical-benchmarks-v1 \
  JACOBIAN_ENABLED=1 EVAL_EXECUTE=1 \
  RUNTIME_SNAPSHOT="$RUNTIME_SNAPSHOT"
```

Use `TASKS=graph-counterexample` for a small smoke run. The treatment run uses
only `benchmarks/config/jacobian.mcp.json`. Task TOMLs remain agent-agnostic;
the treatment measures the MCP product rather than a bundled prompt or Skill.

### Evaluate tool adoption and task design

Jacobian exposes the canonical `math.find` and `math.run` tools in every
observation run. Use a separate result root for each model, task set, or prompt
condition so evidence from distinct runs cannot overwrite each other:

```sh
make agent-eval \
  DATASET=mathematical-benchmarks-v1 JACOBIAN_ENABLED=1 \
  TASKS=graph-counterexample EVAL_EXECUTE=1 \
  RUNTIME_SNAPSHOT=benchmarks/results/adoption/runtime.json \
  EVAL_ARGS="--job-name adoption --jobs-dir benchmarks/results/adoption"
```

Measure appropriate Jacobian adoption, valid calls, discovery-to-execution
continuation, irrelevant calls, fallback behavior, mathematical correctness,
tokens, and elapsed time. A correct run with no Jacobian calls may show that the
task is too easy or that native tools are a better fit; it does not by itself
show that the tool surface was misunderstood. Use negative-control tasks and
tasks where Jacobian offers a material mathematical affordance. Do not score
adherence to a preferred tool sequence.

### Held-out treatment readiness

The protected held-out runner performs a bounded MCP initialization, catalog,
and capability-description preflight before model execution. A held-out model
run starts only when that record reports infrastructure status `READY`. An
unreachable endpoint, timeout, malformed response, or catalog/policy mismatch
records a classified diagnostic and aborts the treatment before the agent
runs. The local `make agent-eval` observation path above does not produce this
held-out readiness contract.

A successful preflight initially records routing status `AVAILABLE_UNUSED`.
Normalization changes that status to `AVAILABLE_INVOKED` only when the trace
contains a successful Jacobian capability invocation. Infrastructure and
routing are independent: an available service that the agent did not use is
not an unavailable service.

## Run without Jacobian

Use the same shared run conditions:

```sh
make agent-eval DATASET=mathematical-benchmarks-v1 \
  JACOBIAN_ENABLED=0 \
  TASKS=graph-counterexample EVAL_EXECUTE=1
```

`JACOBIAN_ENABLED=0` selects the control job without the Jacobian sidecar or
MCP configuration. In the held-out runner, that condition records
infrastructure status `NOT_CONFIGURED` and routing status `NOT_APPLICABLE`, and
performs no Jacobian probe.

Keep the assurance vocabulary identical in both conditions. In particular,
reserve `VERIFIED` for a result whose operator-authorized independent-checker
record satisfies the task contract and binds the exact task input, claim,
semantics, candidate, scope, certificate format, and checker identity. A correct
manual derivation, self-written check, or source citation may support
mathematical correctness, but does not itself establish `COMPUTED` and is not
`VERIFIED`. A producer result carries only the assurance stated by its operation
contract. Put this rule in any ad hoc prompt that does not already expose the
task's structured submission contract. Otherwise a control answer can reuse
`VERIFIED` as an ordinary English synonym and make the assurance comparison
misleading.

`JACOBIAN_ENABLED=1` selects the MCP-only treatment job and passes Harbor's
`--mcp-config` option. `JACOBIAN_ENABLED=0` selects the control with no Jacobian
MCP server. `JACOBIAN_EVAL_PROXY=1` selects matching
proxy-enabled control/treatment job configs and requires at least one proxy
URL variable. The Makefile also passes Harbor's
`web_search=disabled` agent kwarg explicitly, because the `-a codex` and
`-m <model>` command-line overrides replace the agent block from the job JSON.
The default treatment job does not include the proxy overlay.

The pinned Harbor Codex adapter creates a fresh temporary `CODEX_HOME` for
each trial and copies only the job-declared configuration into it.
Therefore a direct host `codex exec`, even with user configuration partially
disabled, is not a valid control for this protocol: installed host apps or
plugins may still be visible. For first-party evidence, inspect each trial's
`lock.json`: control must have no Jacobian MCP server, while treatment must
contain exactly the declared Jacobian MCP server and no Jacobian Skill.

## Docker and Daytona

The checked-in Jacobian evaluation jobs use Docker Compose because Harbor's
local multi-container path needs the `main` container and Jacobian sidecar on
the same network. Daytona is a separate remote execution option and is not
selected by these Makefile commands.

The local image name `jacobian:local` is not pullable by a remote Daytona
worker. A Daytona run therefore needs a published, immutable Jacobian image in
a registry that the worker can access, plus a reachable MCP endpoint or a
Daytona-compatible sidecar configuration. Do not compare a Docker control run
with a Daytona treatment run unless the runtime, image, network, and resource
limits are otherwise held constant.

## Inspect results

Harbor writes results under `benchmarks/results/`. Inspect the summary with:

```sh
uvx --from harbor==0.20.0 harbor view \
  benchmarks/results/mathematical-benchmarks-v1
```

For Jacobian treatment runs, inspect Harbor ATIF together with Jacobian
telemetry. Check capability discovery and descriptions, invocation and
parameter errors, artifact and verification-record flow, repeated or
irrelevant calls, shell/file activity, tokens, time, and completion.
The committed observation jobs explicitly collect Codex's ATIF trajectory from
`/logs/agent/trajectory.json`, allowing the normalizer to measure tool adoption
from manifest-bound evidence. Harbor 0.20 also records its implicit
`/logs/artifacts` publish directory; `empty` is expected there when the agent
publishes no optional files. Failures for explicitly configured artifacts
remain non-conclusions.

For a paired control/treatment run, normalize each condition and then compare
them so task digests, prompts, models, budgets, and job configuration are
checked for drift:

```sh
make agent-eval-validate RESULTS=benchmarks/results/mathematical-benchmarks-v1 \
  JOB=<job-config.json> CONDITION=control \
  OUTPUT=benchmarks/results/normalized-control.json
make agent-eval-validate RESULTS=benchmarks/results/mathematical-benchmarks-v1 \
  JOB=<job-config.json> CONDITION=treatment \
  RUNTIME_SNAPSHOT="$RUNTIME_SNAPSHOT" \
  OUTPUT=benchmarks/results/normalized-treatment.json
make agent-eval-compare \
  CONTROL=benchmarks/results/normalized-control.json \
  TREATMENT=benchmarks/results/normalized-treatment.json \
  OUTPUT=benchmarks/results/comparison.json
```

The comparator rejects unmatched task repetitions or configuration drift and
reports correctness, evidence, scope, assurance, infrastructure, and routing
separately. Missing, unknown, and nonterminal Harbor statuses remain
`UNKNOWN`; they are never normalized into completed runs. See
[Agent evaluation contracts](../reference/evaluations/benchmark-contracts.md)
for the full evidence roles and interpretation boundaries.

Record the git tree, task digests, provider/runtime, model and prompt settings,
raw trace location, and validation actually run. A public workflow result is
regression or observation evidence; it is not held-out causal evidence.

For unstructured pilot answers, validate every claimed assurance level against
its supporting evidence or operation contract and record every unsupported
label as an assurance-vocabulary violation. In particular, `VERIFIED` requires
the complete task-bound record above. Score assurance separately from
mathematical correctness: a correct scalar or proof can still be correct while
its assurance claim is false. Missing checker availability, a timeout, or
failure to obtain a verification record is not evidence that the claim is
false, but it also cannot raise the claim to `VERIFIED`.

## Troubleshooting

If the command exits with `JACOBIAN_MODEL must be exported`, export the model
before invoking Make:

```sh
export JACOBIAN_MODEL='your-model'
```

When `OPENAI_API_KEY` is empty and `~/.codex/auth.json` exists, `agent-eval`
sets `CODEX_FORCE_AUTH_JSON=1` for Harbor automatically. An explicitly set
`CODEX_FORCE_AUTH_JSON` still wins. This avoids Harbor 0.20 selecting an empty
API-key credential instead of an existing ChatGPT login.

If the run stalls at `starting environment`, Docker may be building the task
image or waiting on package installation. Check the Docker build output and
verify proxy reachability from the container before cancelling the trial.

If the treatment agent reports no Jacobian tools, confirm that
`JACOBIAN_ENABLED=1` is set and that the treatment job includes the external
MCP configuration. Inspect the persisted preflight diagnostic first: it
distinguishes endpoint unavailability, malformed protocol responses, and
catalog or policy mismatch from a ready service that the agent did not invoke.
If the preflight cannot reach the sidecar, inspect the Compose network and
ensure the sidecar is listening on `0.0.0.0:8000`.
