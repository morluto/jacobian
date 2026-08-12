---
name: recent-conjecture-evaluations
description: Run source-grounded Jacobian reliability evaluations using recently resolved conjectures as held-out probes. Use for source selection and deduplication, exact input-bound oracles, current-main math.find/math.run contract audits, frozen control/treatment comparisons, observable trajectory scoring, failure attribution, and independent review of another evaluation. Do not use merely to solve a conjecture or create benchmark-specific mathematical helpers.
---

# Recent Conjecture Evaluations

Treat conjectures as probes. The objective is to improve Jacobian from discovery
through final-answer use, not to accumulate solved examples.

This skill owns source selection, deterministic capability audits, optional
control/treatment evaluation, attribution, and repository-action decisions. It
does not create Harbor tasks. Use `harbor-benchmarks` and
`verifier-evaluations` when a selected case is later packaged as a benchmark.

## Select a mode

- **Probe:** run one new source cycle.
- **Review:** independently audit a completed cycle without repeating model calls.
- **Coordinate:** check reservations, duplicates, and ownership before another evaluator proceeds.

Run one cycle by default. Continue looping only when the operator explicitly
requests it. A loop does not authorize unlimited model calls.

## Preserve these invariants

1. Prefer primary sources and record the precise theorem status and date.
2. Bind every oracle, prompt, suite, and tool payload to the intended input.
3. Reconstruct the gold result independently of evaluated model arms.
4. Audit current main deterministically before any model call.
5. Keep `COMPUTED`, `VERIFIED`, source-imported, and unproved claims separate.
6. Treat missing capability, timeout, cancellation, transport failure, and failure to find evidence as non-conclusions.
7. Never attribute a model-authored fallback error to Jacobian when no Jacobian result produced it.
8. Never weaken verification, artifact binding, scope, or assurance to make a case pass.
9. Never merge a PR. Open only a draft PR when the localized-fix gate passes.
10. Never create a benchmark-specific capability from one conjecture family.

## Run the workflow

### 1. Inventory before research

Search saved reports, suites, trajectories, issues, PRs of every state,
branches, and active reservations. Search in layers: exact identifier/title,
artifact, mathematical family, required capability/provider, and suspected root
cause. One exact-source miss does not establish independence.

Run the local helper with positional arguments, then search GitHub separately:

```bash
python .agents/skills/recent-conjecture-evaluations/scripts/search_inventory.py \
  "source or root-cause phrase" outputs benchmarks/results
```

Classify the source as selected, rejected, reserved elsewhere, or completed
duplicate. Read [source-gating.md](references/source-gating.md).

### 2. Establish the source gate

Read enough of the primary source to identify the prior conjecture, exact
resolution status, proof-carrying theorem or construction, finite obligations,
and imported boundaries.

Reject before model calls when the bounded task is illustrative, leaks the
answer, tests literature recall, requires private data, cannot represent the
decisive obligation faithfully, duplicates prior work, or only exercises a
near-match.

### 3. Build the independent oracle

Reconstruct every finite obligation with a clean-room script or exact
derivation. Hash the canonical input. Stop if it disagrees with the primary
source. Label author-supplied code as `SOURCE-SUPPLIED REPLAY`; reserve
`INDEPENDENT ORACLE` for separately implemented or derived evidence.

Record acceptable alternative witnesses and representations. Define what
remains imported or uncertain.

### 4. Audit current main

On a clean current-main worktree, preflight installed providers, authorized
checkers, input bounds, and estimated payload/runtime before materializing a
large artifact. Stop early when the decisive operation is unavailable and the
boundary is already documented.

Then:

1. Search with natural task language.
2. Treat bounded `math.find` search as candidate retrieval, not complete inventory;
   use `capability://catalog` when full installed membership matters.
3. Inspect returned cards and exact IDs for schemas, examples, domains, bounds,
   and provider availability. Treat these as operation facts, not workflow advice.
4. Inspect the producer and independent checker as separate operation contracts.
   A checker-backed result is verified only when the top-level result envelope
   contains a verification-record URI bound to the exact final claim.
5. Execute the frozen input directly.
6. Run the independent checker when installed.
7. Replay relevant malformed, wrong-input, over-bound, and unsupported-domain cases.
8. Run focused deterministic tests.

Persist complete machine-readable discovery and invocation outputs, including
failures, before summarizing them. Hash the input, probe, raw output, and
report. Never support an artifact URI, elapsed time, ranking, or diagnostic
only with terminal scrollback.

If this phase identifies the root cause, do not run model arms merely to
demonstrate it again.

### 5. Decide whether model arms have information value

Read [evaluation-and-scoring.md](references/evaluation-and-scoring.md). Run
paired arms only if every gate passes. Otherwise record a deterministic-only
cycle. Freeze the suite and its digest before execution; use the repository's
evaluation harness to validate the applicable suite shape.

Use fresh isolated contexts and identical settings except for Jacobian
availability. The control must not see Jacobian tools, skills, catalog content,
or routing hints.

### 6. Score observable evidence

Score final correctness, certificate validity, input binding, source fidelity,
scope, completeness, assurance, discovery, contract inspection, execution,
recovery, stopping, elapsed time, token use, and model-visible tool bytes.

Do not claim access to hidden chain-of-thought. Score observable calls, outputs,
retries, concise visible reasoning, and final answers.

### 7. Attribute before acting

Assign each failure to one primary class: Jacobian implementation; tool
interface or contract; discovery or routing; evaluator or telemetry;
infrastructure or transport; model transcription or input binding; model
mathematical reasoning; unsupported capability; or no issue.

Reproduce suspected Jacobian failures directly on current main and search
ownership again. Read [action-policy.md](references/action-policy.md) before
opening an issue or draft PR.

### 8. Publish the cycle record

Write a persistent report using [report-schema.md](references/report-schema.md).
Include rejected and deterministic-only cycles because they prevent duplicate
work. Update the shared reservation ledger. End with a distinct next source
direction, not an unsupported claim that the search space is exhausted.

## Manage cost and stalls

- Prefer deterministic checks.
- Require explicit model-call authorization and cost boundaries.
- Checkpoint after source gate, oracle, current-main replay, each model arm, scoring, and repository action.
- Inspect any phase that produces no checkpoint for 30 minutes.
- Do not retry a cancelled call without new diagnostic evidence.
- Stop collecting examples for an owned root cause once the agreed independence threshold is met.
- Do not use a weaker model merely to manufacture errors; use it only in a frozen comparison with a falsifiable question.
