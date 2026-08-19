# Executable mathematical vocabulary

Jacobian is a demand-driven executable mathematical vocabulary. It exposes
stable, typed mathematical operations that agents can discover and compose. It
does not attempt to enumerate all mathematical knowledge, mirror every backend
API, or encode proof strategies.

The caller owns problem representation, decomposition, sequencing, strategy,
interpretation, and stopping. Jacobian owns the public mathematical contracts,
typed values, execution bounds, and honest result semantics around each move.

## Semantic atomicity

**Atomic means one stable, reusable mathematical postcondition, not a small or
simple implementation.**

A useful mental model is:

```text
given mathematical input X,
return mathematical output Y such that P(X, Y)
```

`X`, `Y`, and `P` should make sense independently of the algorithm, backend,
benchmark, theorem, or surrounding reasoning workflow.

A candidate is near the right semantic boundary when:

1. **It has a mathematical identity.** The contract can be defined without
   mentioning Jacobian, a particular backend, or the motivating problem.
2. **It establishes one postcondition.** The result is one map, predicate,
   invariant, construction, search result, witness, or certificate.
3. **It is strategy-independent.** The result can participate in different
   reasoning strategies and does not prescribe the next operation.
4. **Further splitting loses semantic value.** Smaller pieces would mostly
   expose algorithm state or cheap deterministic projections.

Examples:

| Candidate | Fit | Reason |
| --- | --- | --- |
| Smith normal form | Atomic | One canonical mathematical form. |
| Polynomial factorization | Atomic | One standard mathematical decomposition. |
| Maximum matching | Atomic | One optimization result with a reusable witness. |
| Subgraph embedding | Atomic | One well-defined search relation and witness. |
| Root isolation | Atomic | One mathematical result with explicit enclosure semantics. |
| DFS frontier update | Too low-level | Exposes an implementation step. |
| Extract the first matrix row | Usually too low-level | Cheap projection with little independent leverage. |
| Solve a named conjecture | Too high-level | Encodes the motivating problem and strategy. |
| Analyze a graph and choose the next theorem | Too high-level | Bundles conclusions with planning. |

Algorithmic complexity is not the test. A sophisticated algorithm may implement
one atomic operation, while a tiny helper may still be an implementation detail.

## Discover vocabulary gaps from mathematical work

Grow the vocabulary from observed composition failures rather than from a
top-down inventory of mathematical fields or backend methods:

```text
real mathematical task
  -> composition failure or bespoke-code escape
  -> diagnose the kind of gap
  -> identify the missing mathematical postcondition
  -> test reuse or independent canonicality
  -> consider a public operation
```

When an agent falls back to custom Python, SymPy, FLINT, NetworkX, Z3, or
another system, ask what mathematical fact or object it needed rather than which
helper function it called. Work backward from that result to the stable
mathematical boundary.

For example, bespoke enumeration of simple cycles of a fixed length may expose
a need for a fixed-length-cycle witness. Inspection may reveal a deeper reusable
relation such as finite subgraph embedding. The deeper abstraction is not
automatically better: both may deserve distinct public operations only when they
have distinct discovery intent and useful leverage.

## Diagnose the gap before adding an operation

Not every failed attempt reveals a missing operation.

| Gap | Meaning | Typical response |
| --- | --- | --- |
| Representation | The mathematical object cannot be expressed cleanly. | Improve or add a typed value. |
| Interoperability | Existing operations use incompatible mathematical representations. | Align types or add a domain-owned conversion. |
| Discovery | The operation exists but `math.find` does not surface it. | Improve discovery metadata or examples. |
| Contract | The operation exists but omits needed semantics, witnesses, or evidence. | Repair its request/result contract. |
| Scale/backend | The operation exists but its implementation or bounds are inadequate. | Improve the bounded implementation or backend. |
| Operation | No clean existing composition produces the required mathematical result. | Consider a new public operation. |
| Reasoning | The necessary operations exist but the model does not find the strategy. | Improve reasoning or evaluation rather than the catalog. |

Only a genuine operation gap normally motivates a new public operation.

## What tends to be useful

Useful operations usually turn substantial computation or mathematical
subtlety into typed state that several later moves can consume. Common high-value
shapes include:

- canonical forms and normalizations;
- decompositions and factorizations;
- nontrivial invariants;
- witness-producing searches;
- complete or explicitly bounded enumeration;
- structure-preserving transformations; and
- certified symbolic or numerical computations with explicit guarantees.

Strong candidate signals include repeated bespoke-code escapes, the same
mathematical move recurring in unrelated problems, established concepts in
mature libraries or formalizations, intermediate values that unlock several
downstream compositions, and one narrow contract replacing substantial repeated
custom code.

A backend API is only a source of candidate ideas. Jacobian's public operation
should be stated in mathematical terms and remain meaningful if its private
implementation changes.

## Admission and evaluation

This page explains how candidates are discovered; the
[public operation admission](../reference/public-operation-admission.md)
contract decides whether a candidate belongs in the agent-visible catalog.
Implementation correctness and boundedness are owned by the
[domain operation library](../reference/domain-operation-library.md).

A useful review heuristic is the deletion test: would the operation still
clearly deserve to exist if the motivating benchmark, paper, or conjecture
disappeared? When the answer is unclear, test the candidate on unrelated
mathematical tasks and compare it with existing compositions. Useful evidence
includes cross-problem reuse, fewer bespoke-code escapes, better typed handoffs,
and reliable intermediate witnesses.

No fixed operation count, domain-coverage target, or number of backend wrappers
is a project goal. The vocabulary should grow only when mathematical work
reveals a reusable move that is genuinely missing.
