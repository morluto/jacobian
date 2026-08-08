# Multi-tool mathematical coordination study

This public workflow observation asks whether individually useful Jacobian
operations are discovered and composed into mathematically valid terminal
objects. It transfers the central evaluation lesson from Chen et al.,
*Learning to Coordinate Symbolic Tools*: local symbolic correctness does not
determine global operation choice, ordering, representation handoff, recovery,
or stopping. It does not transfer that paper's synthetic training corpus,
tool-call reward, SFT, or GRPO procedure.

## PR1 preregistration

The frozen
[`multi-tool-coordination-pr1.json`](../../../benchmarks/config/multi-tool-coordination-pr1.json)
selects six existing `mathematical-benchmarks-v1` tasks across graph theory,
algebraic topology, polynomial algebra, optimization, geometry, and integer
linear algebra. Each task receives two independent locally authenticated Codex
rollouts with `gpt-5.4-mini`, medium reasoning, REQUIRED external reasoning
logs, a 600-second process limit, no web search, and no wrong-answer retry.

The task matrix probes graph/set/artifact composition, complex-to-chain and
lattice handoff, polynomial-map candidate checking, rational/positive-definite
slice evidence, coordinate-to-polynomial proof repair, and normal-form
transformation checking. The task instructions remain agent-owned and do not
prescribe a tool sequence.

The normal Harbor observation runner requires Docker, which is unavailable on
the collection host. The preregistered host-local fallback therefore makes no
Harbor execution claim. It exposes only `instruction.md`, `input.json`, and
`submission_schema.json` in a fresh workspace; starts a fresh REQUIRED-log MCP
runtime and state directory per rollout; runs Codex without user configuration
or an API key; and invokes the unchanged task-owned verifier afterward through
the repository's fresh-interpreter virtual-mount harness. This is public
workflow evidence, not a held-out or causal performance comparison.

Only full task-verifier reward is `ACCEPTED`. A completed zero/partial reward is
`REJECTED`. Timeout, model error, verifier infrastructure error, missing model
output, or incomplete execution is `INCONCLUSIVE`, never a negative label.
Tool calls, tokens, and reasoning-log events have reward zero. The analysis
classifies capability discovery, order, intermediate recognition,
representation handoff, checker use, scope/completeness interpretation,
rejection recovery, repeated work, and possible reusable operation gaps only
after all declared rollouts are preserved.

Execution is opt-in and must occur inside a named `tmux` session:

```sh
uv run --locked --with harbor==0.20.0 --with tomli-w==1.2.0 \
  --with jsonschema python -m benchmarks.tooling.multi_tool_coordination_study run \
  --spec benchmarks/config/multi-tool-coordination-pr1.json \
  --output benchmarks/results/multi-tool-coordination-pr1 --execute
```

## PR1 frozen observation

The bounded batch ran once in tmux session `coord-pr1-batch` from clean source
revision `4eaf525a136e4473ddbc015a1d6a94aa0f3dd885`. It completed all 12 planned
rollouts over six task groups: three were accepted and nine rejected, all 12
reasoning protocols completed, and one post-run extraction error was retained.
The immutable manifest binds 196 artifacts. The independent adjudication is
recorded in
[`multi-tool-coordination-pr1-adjudication.json`](../../../benchmarks/config/multi-tool-coordination-pr1-adjudication.json).

Repeated findings included capability-discovery misses, artifact or
representation handoff failures, named-scalar and multiplier-direction binding
errors, undiscovered checkers, false certification after producing an
unverified candidate, and recoverable reasoning-protocol errors. The batch
also contained one accepted multi-operation composition, two safe assurance
downgrades, and one successful repair after checker rejection. Three rejected
answers exposed verifier lexical or directional overconstraints rather than
invalid mathematics; they are retained as contract findings and are not
relabelled.

## PR1 benchmark slice

[`multi-tool-coordination-v1`](../../../benchmarks/datasets/multi-tool-coordination-v1/README.md)
packages four deterministic, hand-auditable Harbor tasks derived from those
observations: graph set distance, a cycle-lattice certificate with alternate
witness support, rational named-scalar binding plus Sylvester replay, and an
explicitly directed polynomial proportionality identity. Hidden Oracle
solutions are absent from the runtime environment. Standard-library clean-room
verifiers independently check the terminal object, frozen input, evidence
bytes, scope, completeness, and assurance, and reject false `VERIFIED` claims.

This is an evidence-bearing pilot, not a calibrated comparison suite. PR1
changes no product capability. Expansion, mixed-difficulty calibration, and
the immutable pre-treatment freeze remain PR2 work.
