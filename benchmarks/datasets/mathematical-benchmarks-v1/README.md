# Jacobian mathematical-benchmarks-v1

This Harbor dataset contains 115 fixed hidden-runtime mathematical tasks for
Oracle validation and optional workflow observation. Each task bundle is a direct child of
this directory and retains separate agent, Oracle, and verifier containers.

`suite.toml` owns stable dataset policy. Each `members/<task>.toml` record owns
membership, provenance, environment, and verifier-contract metadata. Immutable
snapshot locks own frozen corpus identity; no mutable publication manifest is
committed here. The exact-task Oracle contract gate is:

```sh
make harbor-oracle-task DATASET=mathematical-benchmarks-v1 \
  TASKS="graph-counterexample"
```

Use `make harbor-check` and an explicitly scoped `make harbor-oracle` only for
shared Harbor tooling, schemas, registry, suite policy, or other control-plane
changes. Pass `TASKS="..."` for a bounded Oracle; `FULL=1` is reserved for an
intentional complete dataset sweep. Focused commands require explicit task IDs
and never fall back to all tasks.

The standalone observation job uses the Jacobian MCP sidecar. Use it to inspect
how a model applies the public tools; Harbor retains the resolved config and
task lock, trajectory, verifier result, submitted artifacts, and MCP log for
review. A one-attempt run is:

```sh
JACOBIAN_MODEL=gpt-5.6-luna make agent-eval \
  DATASET=mathematical-benchmarks-v1 TASKS=graph-counterexample \
  EVAL_EXECUTE=1 EVAL_ATTEMPTS=1 EVAL_REASONING_EFFORT=high
```

Set `EVAL_ATTEMPTS` for more rollouts. Harbor loads the task bundles from this
dataset directory and applies the task-name filter; Jacobian does not render or
rewrite a Harbor task selection. The MCP endpoint is supplied through Harbor's
external MCP configuration, not through task TOMLs.

For a paired control/treatment run, use the same task filter and model in both
jobs:

```sh
# Control: no Jacobian sidecar or MCP config.
JACOBIAN_MODEL=gpt-5.6-luna make agent-eval DATASET=mathematical-benchmarks-v1 \
  JACOBIAN_ENABLED=0 TASKS=graph-counterexample EVAL_EXECUTE=1

# Treatment: Jacobian sidecar and MCP config.
JACOBIAN_MODEL=gpt-5.6-luna make agent-eval DATASET=mathematical-benchmarks-v1 \
  JACOBIAN_ENABLED=1 \
  TASKS=graph-counterexample EVAL_EXECUTE=1
```

The task bundles and Harbor task digests must be identical between these jobs.
All Jacobian treatments reach the local sidecar through
`http://127.0.0.1:8000/mcp`, independent of external networking. Set
`JACOBIAN_EVAL_PROXY=1` only to apply the same optional upstream egress proxy
to both jobs; only the treatment adds the Jacobian sidecar and MCP config. This
paired setup is for workflow comparison; the public dataset is not held-out
evidence. For a fresh task-image pull behind a proxy, follow the opt-in Buildx
builder setup in [Run agent observations](../../docs/run-agent-evaluations.md).

To evaluate the canonical `math.find` and `math.run` surface, keep each model,
task set, and prompt condition in a separate result root:

```sh
make agent-eval \
  JACOBIAN_MODEL=gpt-5.6-luna DATASET=mathematical-benchmarks-v1 JACOBIAN_ENABLED=1 \
  TASKS=graph-counterexample EVAL_EXECUTE=1 \
  EVAL_ARGS="--job-name adoption --jobs-dir benchmarks/results/adoption"
```

Inspect appropriate adoption, argument validity, discovery-to-execution
continuation, irrelevant calls, native-tool fallback, correctness, tokens, and
elapsed time. A correct no-call run may mean the task does not require a
Jacobian affordance, so include negative controls and tasks that discriminate
tool adoption. Public-suite observations remain directional workflow evidence,
not a causal performance claim.

Jacobian-enabled jobs collect both Codex ATIF and the Jacobian sidecar's MCP
runtime log. The runtime log is authoritative for `math.find` and `math.run`
counts and failed operation attempts; the normalizer does
not infer executions from JavaScript source text. A missing configured trace or
sidecar log makes the observation incomplete. Control jobs collect ATIF only
because they do not start the Jacobian sidecar.

Five tasks have an operator-authorized verification record and may accept
`VERIFIED`; the remaining tasks are capped at `COMPUTED`. A wrong result or an
unsupported certification claim forces reward to zero. These are workflow
observations, not a causal performance benchmark. The committed control and
treatment jobs document a paired workflow setup, but the public suite is not
held out and cannot support a causal performance claim.
