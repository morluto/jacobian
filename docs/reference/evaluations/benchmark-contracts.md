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

Experiments expose the public MCP surface they measure. The agent decides
whether to use Jacobian, whether discovery is useful, which operation to run,
and when to stop; a control condition may expose no Jacobian server. Do not add
product-facing runtime state machines, alternate assurance schemas, or
lifecycle APIs for a study. A reusable benchmark adapter stays a small consumer
of Harbor trial artifacts and the public MCP surface, never a second planner.

## Generated output

Task bundles, member records, and immutable snapshot locks are reproducibility
anchors and remain tracked. Run outputs, collected traces, reports, and other
regenerable evidence belong under ignored `benchmarks/results/` or an external
artifact store. Do not mix generated study output with a runtime or benchmark
source refactor; publish it separately when it is needed for review.
