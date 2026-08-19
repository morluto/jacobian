# Public mathematical operation admission

[Documentation home](../index.md)

- Status: Current catalog-maintenance contract
- Shared admission policy: `src/jacobian/catalog/admission.py`
- Owner-local decisions: `src/jacobian/math/**/_admission.py`

The public `math.find` / `math.run` catalog is a curated basis of mathematical
operations, not an inventory of every callable helper in `jacobian.math` or in
an installed backend. Every candidate declaration must have exactly one
owner-local admission decision before it can enter the catalog. Catalog
construction fails closed when the candidate inventory and composed decision
ledger disagree.

Before applying these gates, identify the reusable gap. Show why the current
public operations and shared mathematical values do not cleanly provide the
required result, and why the proposed postcondition is independently canonical
or reusable beyond the motivating workflow. A discovery, representation,
interoperability, contract, backend, scale, or reasoning failure is not by
itself evidence for a new public operation.

## Admission gates

A public operation must satisfy every gate:

1. It exposes one stable mathematical map, predicate, invariant, construction,
   search, or check rather than a problem-solving workflow.
2. The caller retains representation, decomposition, sequencing, proof
   strategy, and stopping decisions.
3. It returns a reusable typed value, witness, or certificate rather than a
   report or suggested next action.
4. It is exact and bounded, or its result has explicit typed
   `INCOMPLETE`, `UNKNOWN`, or `TRUNCATED` semantics.
5. Its mathematical identity is durable and independent of a benchmark,
   conjecture, theorem instance, or current model behavior.
6. It supplies material computation or reliability leverage over ordinary
   model-authored Python.
7. It is not merely a cheap deterministic projection of another public result.
   Useful projections normally belong only in the native API.
8. Its schema contains no benchmark constants, theorem-specific answer shape,
   or frozen research workflow.
9. It has a distinct discovery intent and does not create a near-duplicate
   result that degrades retrieval.

Passing schema validation, having tests, or wrapping a maintained library does
not by itself satisfy these gates.

## Decisions

The ledger uses five decisions:

| Decision | Catalog effect | Required disposition |
| --- | --- | --- |
| `KEEP` | Public | Preserve the operation ID and contract. |
| `NATIVE_ONLY` | Excluded | Keep the useful deterministic helper under an explicit supported `jacobian.math` symbol. |
| `SPLIT` | Excluded | Do not expose the aggregate; admit smaller outcomes only after independent evidence establishes their discovery intent and leverage. |
| `DROP` | Excluded | Retain no supported public interface solely for compatibility or coverage. |
| `CONTRACT_FIX` | Excluded | Repair the named correctness defect and add an adversarial regression, then reclassify the operation before publication. |

Each mathematical domain's `_admission.py` module is the authority for its
current decisions; `src/jacobian/catalog/admission.py` owns the shared policy
types and fail-closed validation. A renamed or materially changed candidate
needs a fresh decision; do not preserve a public operation solely because an
earlier version was admitted.

Consumers should discover against the current catalog. A `NATIVE_ONLY` row's
`native_symbol` names its supported `jacobian.math` replacement; a `DROP` row
has no compatibility operation.

## Review procedure

For a catalog-changing pull request:

1. Compare the candidate against nearby IDs, native symbols, input and output
   types, and discovery wording.
2. Record one decision and a concrete mathematical rationale in the owning
   domain's `_admission.py` module.
3. For `NATIVE_ONLY`, name an importable callable whose containing public
   module includes it in `__all__`.
4. For bounded search, test both a complete result and the applicable
   incomplete or truncated path. Missing witnesses and exhausted budgets are
   never negative mathematical conclusions.
5. Regenerate the schema snapshot and run the catalog, native-API, and owning
   mathematical tests.

The owner-local decision ledger is source review data for constructing the
immutable public catalog; it is not a runtime recommendation or planning layer.
