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

## PR2 label-free calibration contract

[`multi-tool-coordination-pr2-calibration.json`](../../../benchmarks/config/multi-tool-coordination-pr2-calibration.json)
freezes 12 candidates before collecting any PR2 label. It combines the four
PR1 cross-domain tasks with eight existing `symbolic-coordination-v1` cases:
valid inverse, near miss, one-direction evidence, Keller-only evidence,
collision found, complete grid exhaustion, timeout non-conclusion, and semantic
equivalence. No fixture is copied or rewritten; every task's Harbor digest,
public input bundle, and clean-room verifier bundle are rebound at execution.

Each candidate receives exactly two independent `gpt-5.4-mini` medium runs
with the PR1 prompt, REQUIRED reasoning log, 600-second timeout, no web search,
and no wrong-answer retry. Only a task with exactly one accepted and one
rejected labelled rollout is eligible. Eligible tasks remain in declared
order, at most six are selected, and the target is at least four. Wilson 95%
intervals are descriptive only. Timeout, model error, incomplete reasoning,
missing output, and verifier infrastructure failure are inconclusive.

If fewer than four tasks qualify, the contract permits one separately
preregistered extension of at most six new candidates; it does not launch that
extension automatically. No prompt, model, threshold, task, or product tool
may change in response to calibration labels. The selected set must be frozen
in a later commit before any product treatment.

Execution remains explicit and tmux-only:

```sh
uv run --locked --with harbor==0.20.0 --with tomli-w==1.2.0 \
  --with jsonschema python \
  -m benchmarks.tooling.multi_tool_coordination_calibration run \
  --spec benchmarks/config/multi-tool-coordination-pr2-calibration.json \
  --output benchmarks/results/multi-tool-coordination-pr2-calibration \
  --execute
```

## PR2 initial calibration result

The frozen batch ran once in tmux session `coord-pr2-calibration` from clean
source revision `a33d6cd66f431cf1d904fcc719f24383869d4bd4`. All 24 planned
rollouts completed: 21 were accepted, three were rejected, all reasoning logs
completed, and no outcome was inconclusive. The immutable manifest binds 375
artifacts. Its SHA-256 digest is
`b65f5a32438e44dbc58af7f1323f3d38d86a703df41a1ab8daa37044010944bb`.
The independent per-trajectory audit is recorded in
[`multi-tool-coordination-pr2-calibration-adjudication.json`](../../../benchmarks/config/multi-tool-coordination-pr2-calibration-adjudication.json).

Only `symbolic-coordination-semantic-equivalence-01` met the frozen one
accepted/one rejected rule. Both directed-proportionality rollouts made the
same fixed-parameter inversion through different operation routes. The mixed
semantic-equivalence task separated a failed recovery from noncanonical sparse
input from a successful exact residual replay. Ten post-run typed-state
extractions failed because certificate metadata was misclassified as a
candidate scalar; the raw traces and task-owned verifier labels remain intact.

Because one selected task is below the target of four, the initial contract
authorizes exactly one separately preregistered extension. The extension is
not a retry: it freezes six previously unrun tasks in
[`multi-tool-coordination-pr2-calibration-extension.json`](../../../benchmarks/config/multi-tool-coordination-pr2-calibration-extension.json),
keeps every model, prompt, timeout, verifier, and label rule unchanged, and
binds the initial manifest and summary digests. It includes rational inverse
and near-miss encodings, a three-variable one-direction case, and three
distinct unused semantic-equivalence encodings. No second extension is
authorized, even if the combined target remains unmet.

The extension is opt-in and must use its own named tmux session:

```sh
uv run --locked --with harbor==0.20.0 --with tomli-w==1.2.0 \
  --with jsonschema python \
  -m benchmarks.tooling.multi_tool_coordination_calibration run \
  --spec benchmarks/config/multi-tool-coordination-pr2-calibration-extension.json \
  --output benchmarks/results/multi-tool-coordination-pr2-calibration-extension \
  --execute
```

## PR2 extension result and frozen comparison

The one authorized extension ran once in tmux session
`coord-pr2-calibration-extension` from clean source revision
`9bf71ae053e0cdaa9cd981c9ee875536f24728d0`. All 12 rollouts completed:
11 were accepted, one was rejected, every reasoning protocol completed, and
no result was inconclusive. The 185-artifact manifest digest is
`ac1ac4b541c7771fa5ee2a8aeb302bad1b159f3fffe23e9bc90206cb833bd49a`.
The complete audit is recorded in
[`multi-tool-coordination-pr2-calibration-extension-adjudication.json`](../../../benchmarks/config/multi-tool-coordination-pr2-calibration-extension-adjudication.json).

Only `symbolic-coordination-semantic-equivalence-04` was mixed. Its rejected
rollout recovered from an invalid noncanonical request and obtained a correct
VERIFIED inverse result, but then duplicated scope, completeness, and
limitations inside the terminal result object. The clean-room verifier
correctly rejected that representation handoff. Other extension traces
repeated fail-closed sparse-map ordering or duplicate-term errors, but usually
recovered or completed the exact terminal object locally.

Combining both calibration stages leaves two mixed tasks, below the target of
four. The extension is exhausted and no further candidate run is authorized.
The exact two-task evaluation set and a five-repetition-per-condition
exploratory comparison contract are frozen in
[`multi-tool-coordination-pr3-frozen-comparison.json`](../../../benchmarks/config/multi-tool-coordination-pr3-frozen-comparison.json).
Both tasks are semantic-equivalence cases, so a later before/after result must
not be presented as broad cross-domain performance or as a causal claim. PR2
contains no product capability change.
