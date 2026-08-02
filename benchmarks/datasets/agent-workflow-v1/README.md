# Jacobian agent-workflow-v1

This Harbor dataset contains fixed mathematical workflows for Oracle validation
and agent observation. Each task bundle is a direct child of
this directory and retains separate agent, Oracle, and verifier containers.

`suite.toml` owns membership and contract metadata. `dataset.toml` is generated
from Harbor-computed task digests. The Oracle contract gate is:

```sh
make harbor-oracle DATASET=agent-workflow-v1
```

The standalone observation job uses the Jacobian MCP sidecar and is selected
with:

```sh
make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=1 EVAL_EXECUTE=1
```

Use `TASKS=graph-counterexample` for a small run. Harbor loads the task bundles
from this dataset directory and applies the task-name filter; Jacobian does not
render or rewrite a Harbor task selection. The MCP endpoint is supplied through
Harbor's external MCP configuration, not through task TOMLs.

For a paired control/treatment run, use the same task filter and model in both
jobs:

```sh
# Control: no Jacobian sidecar or MCP config.
make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=0 TASKS=graph-counterexample EVAL_EXECUTE=1

# Treatment: Jacobian sidecar and MCP config.
make agent-eval DATASET=agent-workflow-v1 \
  JACOBIAN_ENABLED=1 \
  TASKS=graph-counterexample EVAL_EXECUTE=1
```

The task bundles and generated task digests must be identical between these
jobs. Both jobs use the same optional proxy overlay; only the treatment adds
the Jacobian sidecar and MCP config. This paired setup is for
workflow comparison; the public dataset is not held-out evidence.

Five tasks have an operator-authorized verification record and may accept
`VERIFIED`; the remaining tasks are capped at `COMPUTED`. A wrong result or an
unsupported certification claim forces reward to zero. These are workflow
observations, not a causal performance benchmark. The committed control and
treatment jobs document a paired workflow setup, but the public suite is not
held out and cannot support a causal performance claim.

Version 1.2.0 adds the independently reviewed finite-magma and
well-total-domination countermodel cases. Both remain public workflow evidence;
neither is a held-out capability-benefit case.
