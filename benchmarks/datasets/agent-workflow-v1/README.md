# Jacobian agent-workflow-v1

This Harbor dataset contains 26 fixed Jacobian-enabled mathematical workflows
for Oracle validation and agent observation. Tasks are organized as
`tasks/mathematical-sciences/<field>/<task-name>/` and retain separate agent,
Oracle, and verifier containers.

`suite.toml` owns membership and contract metadata. `dataset.toml` is generated
from Harbor-computed task digests. The Oracle contract gate is:

```sh
make harbor-oracle DATASET=agent-workflow-v1
```

The observation job uses the Jacobian MCP sidecar and is selected with:

```sh
make agent-eval DATASET=agent-workflow-v1 EVAL_EXECUTE=1
```

Five tasks have an operator-authorized verification record and may accept
`VERIFIED`; the remaining tasks are capped at `COMPUTED`. A wrong result or an
unsupported certification claim forces reward to zero. These are workflow
observations, not a causal performance benchmark: the dataset has no control
condition or randomized pairing.

Version 1.1.0 adds the independently reviewed finite-magma and
well-total-domination countermodel cases. Both remain public workflow evidence;
neither is a held-out capability-benefit case.
