---
name: learn-from-math-agent-trajectories
description: Review mathematical agent trajectories for evidence-backed Jacobian improvements; do not resume solving.
---

# Learn from Math Agent Trajectories

Extract reusable Jacobian lessons from a completed or paused investigation;
do not resume solving the problem. A correct answer can expose workflow defects,
and an unsuccessful search can reveal useful mathematical vocabulary.

## Establish the evidence

Record the intended and actual outcome, stopping condition, transcript coverage,
and available revision/catalog context. Use observable sources, calls, code,
artifacts, corrections, and final claims. Narration alone is not execution
evidence; current-main capabilities were not necessarily available in the trace.

Reconstruct decisions that changed correctness, cost, progress, or confidence.
For a finding that depends on mathematical claims, numerical or symbolic work,
solver semantics, or bespoke code, consult
[mathematical evidence](references/mathematical-evidence.md). Preserve decisive
claims and later corrections, with their hypotheses and evidence scope.

## Attribute the lesson

Distinguish working capabilities, environment limitations, discovery/selection,
execution friction, representation/interoperability, contract/scale defects,
missing operations, handoff failures, and caller reasoning. Compare needed
postconditions with the session-visible catalog when available; check current
source before proposing new work. Handwritten code and tool non-use are leads,
not automatic evidence of missing operations.

Separate a reusable operation gap from public-catalog admission. An existing
postcondition with a narrow envelope is a scale/backend question. A convenience
or theorem-specific assembly does not become a public operation solely because
it occurred in the trace. Use the
[admission contract](../../../docs/reference/public-operation-admission.md)
when proposing publication.

Route only when the requested follow-up needs a deeper workflow:

- `evaluate-mcp-tool-adoption` for controlled availability, discovery, or selection;
- `audit-mcp-tool-friction` for problems after selecting a tool;
- `audit-public-operation-contracts` for a particular mathematical contract; or
- `recent-conjecture-evaluations` for a new held-out reliability probe.

An unresolved conjecture is generally unsuitable as an evaluation oracle.
Extract frozen, independently checkable finite obligations when proposing an
evaluation, and preserve contamination boundaries.

## Report actionable learning

For each material finding, give the source evidence, implication, proportionate
repair, and uncertainty. Include a compact claim/correction ledger when it helps
explain a changed conclusion; include ownership and success criteria when
proposing implementation. Search narrowly for an existing issue before suggesting
a new one. External mutations require user authorization.

Prefer discovery, contract, representation, or scale repairs when they explain
the failure. Update skills only for reusable decision guidance and product docs
only for durable public behavior. An isolated agent slip may need no repository
change. Stop when the requested trace is accounted for and the supported lessons
and remaining proof gaps are clear; report coverage rather than implying that
the underlying mathematical problem is solved.
