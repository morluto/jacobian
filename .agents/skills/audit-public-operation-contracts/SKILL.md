---
name: audit-public-operation-contracts
description: Audit new or materially changed Jacobian mathematical operations for public-domain mismatches, hidden work expansion, lossy exact results, source-unbound conclusions, and producer-consumer incompatibility. Use for operation-contract reviews and adjacent-domain searches; do not use for general repository, CI, MCP-discovery, or security audits.
---

# Audit Public Operation Contracts

Audit whether a Jacobian operation is a bounded, truthful, composable
mathematical instrument. Treat the request model, native value, kernel or
backend adapter, result model, declaration, examples, and downstream consumers
as one contract.

Before auditing, read the current relevant sections of:

- `AGENTS.md`;
- `docs/reference/domain-operation-library.md`;
- `docs/reference/testing-strategy.md`; and
- `docs/reference/mathematical-backends.md` when a backend is involved.

Use those files as the policy authority. Do not copy their general rules into
the report.

## Establish the audit boundary

Record the revision, operation IDs, owner domain, changed paths, backend and
version when relevant, and whether the request is audit-only or also authorizes
implementation. Local investigation never authorizes commits, pushes, issue or
PR changes, comments, or thread resolution.

Map each operation end to end:

```text
public request model
  -> canonical domain value and admission
  -> native kernel or private backend adapter
  -> exact result conversion and invariant validation
  -> canonical result and downstream consumers
```

Inspect generated schema, declaration text, examples, and MCP-visible errors
when the public projection is in scope. Do not infer the public contract from
the implementation alone.

## Complete the contract preflight

Fill the review artifact from `domain-operation-library.md`. For every produced
mathematical value, additionally answer:

- What owner-defined canonical type is returned?
- Which downstream operations consume it?
- Can its serialized value enter those consumers unchanged?
- Does it retain parent, presentation, ordered axes, ambient dimension, and
  normalization where meaningful?
- What context survives empty, zero, identity, and other degenerate values?
- Is a decision or certificate bound to the exact source value it concerns?
- Can validation replay its defining relation within the admitted work bound?

Classify each output as a canonical value, source-bound result, or display
projection. A display projection must not masquerade as a composable value.

## Probe the recurrent failure classes

Use small deterministic reproductions before broad mutation. Check for:

- a request accepted beyond the kernel or backend's mathematical domain;
- input, intermediate, algorithmic-work, or exact-result growth omitted from
  admission, especially exponentiation, dense products, closure, and replay;
- backend units, multiplicities, generators, witnesses, parents, or axes lost
  during conversion;
- exact properties inferred from the shape of lossy output;
- a result model that accepts a forged conclusion, certificate, or derived
  value independently of its source;
- parallel producer and consumer representations that require caller-side
  reconstruction;
- empty or singular values that lose their ambient mathematical context;
- implicit coercion across rings, fields, parents, presentations, or axes;
- implementation semantics that disagree with title, description, scope,
  method, examples, or result field names; and
- a private backend imported or exposed at the public namespace boundary.

When a finding depends on a maintained library, verify its current documented
domain and semantics against official documentation and the pinned version.
Generated documentation summaries are discovery aids; confirm consequential
claims in source, tests, or release documentation.

## Search for the shared mechanism

After proving a defect, search the owner domain, shared helper, and every caller
of that helper for the same mechanism. Keep the search causal: do not turn one
finding into an unbounded repository audit. Report each adjacent candidate as
confirmed, disproved, or untested.

Prefer a root repair at the owning layer:

- admission for unsupported or excessive requests;
- canonical value ownership for composition failures;
- adapter conversion for backend information loss;
- result validation for source authenticity and reconstruction; or
- declaration/schema text for a public semantic mismatch.

Do not use result validation to compensate for overbroad admission, and do not
add source-text lint rules for mathematical properties.

## Require discriminating evidence

For a confirmed bug, first preserve a focused behavioral regression that fails
on the base for the intended reason when feasible. Then use the smallest
relevant evidence:

- accepted and immediately rejected boundary cases;
- reconstruction or defining-identity properties;
- producer -> serialization -> consumer closure, including a degenerate value;
- independent mutation of source and conclusion fields;
- a bounded independent oracle or metamorphic property for mathematical logic;
- catalog invocation for public projection changes; and
- the owner lane named in `CONTRIBUTING.md`.

Examples are necessary evidence for discoverability, not proof of mathematical
correctness. Generic fuzzing can expose validation gaps but cannot replace a
domain invariant or independent oracle.

## Report the audit

Lead with the conclusion. For every confirmed finding include the affected
operation, public claim, minimal reproduction, observed result, mathematical or
architectural invariant violated, shared root mechanism, affected siblings,
smallest repair, and regression evidence. Separate confirmed facts from
hypotheses and list meaningful proof gaps.

If implementation was authorized, preserve unrelated work, make focused
changes with their tests, run the owning validation, and report only evidence
that actually ran. Do not perform external mutations without explicit
authorization.
