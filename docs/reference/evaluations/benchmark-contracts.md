# Benchmark contracts

[Documentation home](../../index.md) · [Tool surface](../tools.md)

Harbor benchmarks are external evaluation products, not Jacobian runtime
features. A task owns its agent-visible instruction, hidden Oracle material,
verifier, pinned environment, and task digest under `benchmarks/datasets/`.
The evaluation author owns task selection, scoring, and any independent
validation needed by that task.

Keep the boundary sharp: a benchmark may call `math.find` and `math.run`, but
it must pass complete bounded inputs and use returned typed values. It must not
assume a Jacobian workspace, artifact URI, persistence layer, checker registry,
or verification record. Freeze task and environment digests, keep hidden
material out of the agent environment, and report the task's limitations with
its score.
