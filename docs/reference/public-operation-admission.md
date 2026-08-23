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
itself evidence for a new public operation. See
[Executable mathematical vocabulary](../explanation/executable-mathematical-vocabulary.md)
for the semantic-atomicity test and gap-diagnosis methodology.

Do not confuse a stable mathematical postcondition with one release's admitted
execution envelope. If an existing operation has the right semantics but uses
an unnecessarily coarse or narrow limit, classify the problem as a
scale/backend gap and widen that contract when its work, intermediate growth,
and result remain bounded. A new operation must not exist solely to bypass an
arbitrary cap on a composable primitive. Follow the
[boundedness proof](domain-operation-library.md#boundedness-proof) and prefer
the quantities that actually control the kernel or exact output.

## Execution-envelope review

Keep the public mathematical postcondition as broad as its semantics permit.
An implementation or backend limit describes the current admitted execution
envelope; it does not redefine the mathematical objects to which the operation
applies. Before adopting a small fixed input cap, complete this review:

1. Identify the quantities that control work, intermediate growth, and exact
   output. Use quantities such as operand digits, coefficient height, degree,
   terms, matrix dimensions, candidate count, witness count, or predicted
   serialized size rather than a convenient coarse input field.
2. Compare exact algorithm and representation regimes. Consider sparse,
   factored, modular, symbolic, or implicit values before requiring an expanded
   result, and separate decision, first-witness, and complete-profile contracts
   when their output obligations differ.
3. State whether the accepted source is materialized, succinct, generated, or
   oracle-backed. Identify every expansion the kernel performs, prove that
   admission can bound it before execution, and check whether an apparently
   equivalent compact representation changes the complexity class or output
   obligation. Representation is part of the admitted domain, not an adapter
   detail.
4. Research a maintained specialist backend before writing a custom kernel or
   retaining a restrictive pure-Python path. FLINT, GMP, and Arb are relevant
   examples for exact integer, polynomial, matrix, and rigorous ball
   computation; they are not mandatory when another maintained backend better
   fits the operation.
5. Define preflight admission from the selected algorithm's work,
   intermediate, memory, and result bounds. Large scalar inputs should remain
   admissible when those derived quantities and the returned value are small.
6. Document any remaining fixed ceiling as a conservative fallback. State
   whether it is a mathematical, representation, backend, or currently
   uninvestigated limit, and identify the evidence needed to raise it.
7. Test accepted and rejected boundaries, algorithm or representation
   crossover points, and realistic source-backed cases. Use defining
   invariants or an independent oracle to show that every selected regime has
   the same public semantics.

A timeout, cancellation, resource exhaustion, or backend `UNKNOWN` result is
an execution outcome, never a negative mathematical conclusion. If the public
result has no typed incomplete or unknown state, admission must reject the
request before execution whenever completion cannot be bounded. Wall time
remains a safety net rather than the definition of the mathematical domain.

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
10. Its admitted representation does not hide an unbounded expansion or a
    materially different computational problem. The request and preflight name
    and bound any expansion before execution.

A named technique does not justify a second operation when it has the same
mathematical input, output, and defining relation as an existing operation.
Record the technique as discovery vocabulary or a private kernel. Require a
distinct reusable result, witness, decomposition, or certificate for separate
admission.

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
current decisions and exports one `REGISTRATION` binding its candidate `TOOLS`
to those decisions. The packaged `_admission.py` path is the explicit
publication marker; catalog construction discovers those owner modules in
deterministic path order and does not load external entry points or plugins.
`src/jacobian/catalog/admission.py` owns the shared policy types and fail-closed
validation. A renamed or materially changed candidate needs a fresh decision;
do not preserve a public operation solely because an earlier version was
admitted.

Consumers should discover against the current catalog. A `NATIVE_ONLY` row's
`native_symbol` names its supported `jacobian.math` replacement; a `DROP` row
has no compatibility operation.

## Review procedure

For a catalog-changing pull request:

1. Compare the candidate against nearby IDs, native symbols, input and output
   types, and discovery wording.
2. Complete the execution-envelope review above. Verify that each request
   limit follows from a named representation, work, intermediate-growth, or
   result-size budget; record the algorithm/backend regime and whether a
   sharper bound safely admits materially larger source-backed cases.
3. Record one decision and a concrete mathematical rationale in the owning
   domain's `_admission.py` module.
4. For `NATIVE_ONLY`, name an importable callable whose containing public
   module includes it in `__all__`.
5. For bounded search, test both a complete result and the applicable
   incomplete or truncated path. Missing witnesses and exhausted budgets are
   never negative mathematical conclusions.
6. Regenerate the schema snapshot and run the catalog, native-API, and owning
   mathematical tests.

The owner-local decision ledger is source review data for constructing the
immutable public catalog; it is not a runtime recommendation or planning layer.
