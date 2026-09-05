---
name: audit-mathematical-vocabulary
description: Audit a bounded mathematical slice for missing or unusable Jacobian capabilities, beyond a single-operation review.
---

# Audit Mathematical Vocabulary

Find confirmed gaps in one mathematical slice without turning the audit into a
backend-wrapper wishlist or a claim of complete coverage.

Use the relevant [vocabulary model](../../../docs/explanation/executable-mathematical-vocabulary.md)
and domain references to assess the slice. Consult the
[admission contract](../../../docs/reference/public-operation-admission.md)
when proposing publication or changing an execution envelope. Use
`audit-public-operation-contracts` for a deep review of one operation and
`recent-conjecture-evaluations` for a held-out conjecture evaluation.

## Audit one slice

Choose one bounded slice, such as polynomial elimination. Name its mathematical
boundary and the current-main revision. Do not silently expand it into a whole
domain or repository audit.

Snapshot the relevant catalog facts before researching solutions. Record the
operation IDs, owner domains, canonical inputs and outputs, natural downstream
consumers, defining or reconstruction invariants, admitted bounds, and private
backends. Use the live catalog and current source as authority; search results
alone are not a complete inventory.

Check the slice through four independent lenses:

1. **Source demand.** Use primary sources or established exact tasks to identify
   a finite mathematical obligation. State the needed result without importing
   the source's proof strategy. Literature recall or one bespoke workflow is not
   enough evidence for a public operation.
2. **Composition.** Test whether producer values can enter their natural
   consumers unchanged after serialization. Include the empty, zero, singular,
   or identity case most likely to lose parent, axes, multiplicity, witness, or
   ambient context.
3. **Contract.** Check accepted input, work and intermediate growth, output
   bounds, exact-result reconstruction, source binding, and typed failure
   behavior. Separate the semantic mathematical domain from the current
   execution envelope, and identify the quantities that actually control work
   and result size. A timeout, unavailable backend, or incomplete search is not
   a mathematical conclusion.
4. **Backend feasibility.** Check maintained exact backends for an appropriate
   kernel only after establishing mathematical demand. Compare compact exact
   representations and algorithm regimes before treating a small fixed cap as
   necessary. Backend availability is evidence of feasibility, not evidence
   that Jacobian needs another public operation.

Prefer deterministic inspection and direct operation calls. Use an independent
oracle when the finding depends on a mathematical value. Do not run model
comparisons unless deterministic evidence cannot distinguish discovery from
reasoning failure.

## Classify before proposing work

Assign each finding one primary class:

- `operation`: the reusable mathematical postcondition is absent;
- `representation`: the mathematical value cannot be expressed canonically;
- `interoperability`: existing producers and consumers use incompatible values;
- `discovery`: the operation exists but natural search does not surface it;
- `contract`: the operation exists but its public semantics or evidence are
  insufficient;
- `scale/backend`: the operation exists with the needed postcondition, but a
  coarse admission proxy, expanded representation, algorithm regime, or
  bounded implementation is the limiting factor;
- `reasoning`: the needed vocabulary exists and the remaining failure is
  strategy selection or mathematical reasoning; or
- `no gap`: the investigated path is already supported or does not justify a
  product change.

Before opening anything, search local reports and live issues and pull requests
of every state for the mathematical family, operation IDs, and root mechanism.
Treat an owned finding as ownership evidence, not a new issue. External issue,
comment, or pull-request mutations still require explicit authorization.

A new gap is ready to file only when it has a stable mathematical
postcondition, a plausible bounded exact implementation, useful composition
with canonical values, concrete evidence beyond a single trace or backend API,
and no existing owner. Lead the issue with the user or mathematical need, keep
confirmed facts separate from hypotheses, and leave admission or implementation
details open unless the evidence settles them.

When external issue creation is authorized, make the issue implementable rather
than merely descriptive.  Include the smallest discriminating exact fixture,
the defining invariant or an independent oracle, the relevant degenerate or
adversarial case, the preflight quantities that bound work and exact output,
and explicit non-goals.  For a repair, also state the existing owner and the
smallest producer-consumer or public-contract regression.  Do not invent a
backend prescription when the evidence establishes only the postcondition.

Do not file a new operation solely because an existing operation rejects a
large input. First determine whether predicted work, intermediate growth, and
exact output remain small; whether a sparse, factored, modular, symbolic, or
implicit representation avoids expansion; and whether a maintained backend or
different exact algorithm materially widens the envelope. If so, record a
scale/backend or admission gap against the existing postcondition.

## Report the result

Report the slice and revision, evidence coverage, classified findings, ownership,
and meaningful proof gaps. A valid audit may end with `no gap`. Propose another
slice only when continued exploration is relevant to the request.

Persist a report when requested or needed for an ongoing audit campaign. Keep
revision, slice, outcome, and ownership identifiable so later audits can check
coverage; do not maintain a separate database for a few reports.
