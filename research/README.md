# Research records

Research records document capability discovery, evaluation plans, handoffs, and
eventual reports. They are separate from executable Harbor benchmarks:

- `benchmarks/datasets/` contains executable Harbor cases;
- `benchmarks/tooling/` contains reusable Harbor infrastructure; and
- `research/evaluations/` contains non-runnable plans and handoffs.

Records under `research/` are not injected into agent containers and must not
be used as performance evidence without a separately frozen held-out
evaluation. A research record may reference a canonical dataset, but it must
not duplicate task paths, fixtures, Oracle material, or Harbor job inputs.

Evaluation handoffs are kept with the task or immutable snapshot that owns
them. The committed `agent-workflow-v1` lock under `benchmarks/snapshots/`
defines the reproducible task set; task-owned `analysis/gap.json` records carry
optional discovery context without becoming a mutable suite-wide ledger.
