# Run agent evaluations

[Documentation home](../index.md) · [Evaluation reference](../reference/evaluations/evaluation-methods.md)

This guide covers an explicit operator-run of Harbor workflow observations and
paired Jacobian control/treatment runs. Model execution is not a routine
development or pull-request gate. The evaluation roles, assurance rules, and
interpretation boundaries are in the [reference page](../reference/evaluations/evaluation-methods.md).

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
make harbor-check-task DATASET=agent-workflow-v1 TASKS="graph-counterexample"
make harbor-oracle-task DATASET=agent-workflow-v1 TASKS="graph-counterexample"
```

Use the full `make harbor-check` and explicitly scoped `make harbor-oracle`
paths only for shared Harbor tooling, schemas, registry, suite policy, or
other control-plane changes. Pass `TASKS="..."` for a bounded dataset Oracle;
pass `FULL=1` only when a complete dataset sweep is intentional.

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
It renders a private runtime configuration that chains Harbor's transparent
egress controller through the upstream proxy, so Harbor's task allowlist stays
in force. Codex and Jacobian share Harbor's controlled network namespace, and
MCP uses its loopback interface, so local tool traffic never enters the
upstream chain. All jobs also mount the host's standalone Codex executable to
avoid a trial-time package install; override its resolved path with
`JACOBIAN_EVAL_CODEX_BINARY`. Keep machine-specific Clash addresses in local
environment variables rather than committed configuration.

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
export SNAPSHOT_LOCK='benchmarks/snapshots/agent-workflow-v1/6c6d41612502da5486bc23843e027a30cf91398ecf7c749cb8a017c56490707d.lock.json'
export RUNTIME_SNAPSHOT='benchmarks/results/my-run/runtime.json'
mkdir -p "$(dirname "$RUNTIME_SNAPSHOT")"
jq --arg model "$JACOBIAN_MODEL" \
  '{snapshot_id, harbor_version: "0.20.0", model: $model,
    condition: {id: "treatment", role: "PRIMARY_TREATMENT",
                jacobian_enabled: true, reasoning_log_mode: "OFF"}}' \
  "$SNAPSHOT_LOCK" > "$RUNTIME_SNAPSHOT"

make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=1 EVAL_EXECUTE=1 \
  RUNTIME_SNAPSHOT="$RUNTIME_SNAPSHOT"
```

Use `TASKS=graph-counterexample` for a small smoke run. The treatment run uses
`benchmarks/config/jacobian.mcp.json`; task TOMLs remain agent-agnostic.

### Evaluate tool adoption and task design

Jacobian exposes the canonical `math.find` and `math.run` tools in every
observation run. Use a separate result root for each model, task set, or prompt
condition so evidence from distinct runs cannot overwrite each other:

```sh
make agent-eval \
  DATASET=agent-workflow-v1 JACOBIAN_ENABLED=1 \
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
make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=0 \
  TASKS=graph-counterexample EVAL_EXECUTE=1
```

`JACOBIAN_ENABLED=0` selects the control job without the Jacobian sidecar or
MCP configuration. In the held-out runner, that condition records
infrastructure status `NOT_CONFIGURED` and routing status `NOT_APPLICABLE`, and
performs no Jacobian probe.
`JACOBIAN_ENABLED=1` selects the treatment job and passes
Harbor's `--mcp-config` option. `JACOBIAN_EVAL_PROXY=1` selects matching
proxy-enabled control/treatment job configs and requires at least one proxy
URL variable. The Makefile also passes Harbor's
`web_search=disabled` agent kwarg explicitly, because the `-a codex` and
`-m <model>` command-line overrides replace the agent block from the job JSON.
The default treatment job does not include the proxy overlay.

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
  benchmarks/results/agent-workflow-v1
```

For Jacobian treatment runs, inspect Harbor ATIF together with Jacobian
telemetry. Check capability discovery and descriptions, invocation and
parameter errors, artifact and verification-record flow, repeated or
irrelevant calls, shell/file activity, tokens, time, and completion.

For a paired control/treatment run, normalize each condition and then compare
them so task digests, prompts, models, budgets, and job configuration are
checked for drift:

```sh
make agent-eval-validate RESULTS=benchmarks/results/agent-workflow-v1 \
  JOB=<job-name> CONDITION=control OUTPUT=benchmarks/results/normalized-control.json
make agent-eval-validate RESULTS=benchmarks/results/agent-workflow-v1 \
  JOB=<job-name> CONDITION=treatment \
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
[Capability workflow evaluations](../reference/evaluations/benchmark-contracts.md)
for the full evidence roles and interpretation boundaries.

Record the git tree, task digests, provider/runtime, model and prompt settings,
raw trace location, and validation actually run. A public workflow result is
regression or observation evidence; it is not held-out causal evidence.

## Troubleshooting

If the command exits with `JACOBIAN_MODEL must be exported`, export the model
before invoking Make:

```sh
export JACOBIAN_MODEL='your-model'
```

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
