# Jacobian agent-workflow-v1

This Harbor dataset contains fixed Jacobian-enabled mathematical workflows for
Oracle validation and agent observation. Each task bundle is a direct child of
this directory and retains separate agent, Oracle, and verifier containers.

`suite.toml` owns membership and contract metadata. `dataset.toml` is generated
from Harbor-computed task digests. The Oracle contract gate is:

```sh
make harbor-oracle DATASET=agent-workflow-v1
```

The observation job uses the Jacobian MCP sidecar and is selected with:

```sh
make agent-eval DATASET=agent-workflow-v1 EVAL_EXECUTE=1
```

Use `TASKS=graph-counterexample` for a small run. Harbor loads the task bundles
from this dataset directory and applies the task-name filter; Jacobian does not
render or rewrite a Harbor task selection.

Five tasks have an operator-authorized verification record and may accept
`VERIFIED`; the remaining tasks are capped at `COMPUTED`. A wrong result or an
unsupported certification claim forces reward to zero. These are workflow
observations, not a causal performance benchmark: the dataset has no control
condition or randomized pairing.

Version 1.2.0 adds the independently reviewed finite-magma and
well-total-domination countermodel cases. Both remain public workflow evidence;
neither is a held-out capability-benefit case.
