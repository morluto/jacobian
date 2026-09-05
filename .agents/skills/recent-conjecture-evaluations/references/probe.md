# Run one reliability probe

## Run the workflow

### 1. Inventory before research

Search saved reports, suites, trajectories, issues, PRs of every state,
branches, and active reservations. Search in layers: exact identifier/title,
artifact, mathematical family, required operation, and suspected root
cause. One exact-source miss does not establish independence.

Run the local helper with positional arguments, then search GitHub separately:

```bash
python .agents/skills/recent-conjecture-evaluations/scripts/search_inventory.py \
  "source or root-cause phrase" outputs benchmarks/results
```

Classify the source as selected, rejected, reserved elsewhere, or completed
duplicate. Read [source-gating.md](source-gating.md).

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
derivation. Hash the canonical input. If it disagrees with the primary source,
pause model evaluation and investigate the discrepancy within the agreed budget.
Proceed only once it is resolved; otherwise record the disagreement and close
the cycle with that limitation. Label author-supplied code as
`SOURCE-SUPPLIED REPLAY`; reserve
`INDEPENDENT ORACLE` for separately implemented or derived evidence.

Record acceptable alternative witnesses and representations. Define what
remains imported or uncertain.

### 4. Audit current main

For a new probe, use a clean current-main worktree to check installed operations,
claim-checking capabilities, input bounds, and estimated payload/runtime before
materializing a large artifact. When the decisive operation is unavailable and
the boundary is already documented, skip dependent execution and model arms, then complete attribution
and the cycle record.

Then:

1. Search with natural task language.
2. Treat bounded `math.find` search as candidate retrieval, not complete inventory;
   use `operation://catalog` when full installed membership matters.
3. Inspect returned cards and exact IDs for schemas, examples, domains, bounds,
   and backend availability. Treat these as operation facts, not workflow advice.
4. Inspect the producer and independent checker as separate operation contracts.
   A checker verdict must bind the exact claim and semantics; do not expect a
   generic verification record or artifact URI from an ordinary operation.
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

Read [evaluation-and-scoring.md](evaluation-and-scoring.md). Run
paired arms only if every gate passes. Otherwise record a deterministic-only
cycle. Freeze the suite and its digest before execution; use the repository's
evaluation harness to validate the applicable suite shape.

Use fresh isolated contexts and identical settings except for Jacobian
availability. The control must not see Jacobian tools, skills, catalog content,
or routing hints. The treatment exposes the public surface only: the agent
still decides whether discovery is useful, which operation to call, and when to
stop.

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
mathematical reasoning; unsupported operation; or no issue.

Reproduce suspected Jacobian failures directly on current main and search
ownership again. Read [action-policy.md](action-policy.md) before
opening an issue or draft PR.

### 8. Publish the cycle record

Write a persistent report using [report-schema.md](report-schema.md).
Include rejected and deterministic-only cycles because they prevent duplicate
work. Update the shared reservation ledger. End with a distinct next source
direction, not an unsupported claim that the search space is exhausted.
