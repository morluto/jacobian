# Executable mathematical vocabulary

Jacobian is a demand-driven executable mathematical vocabulary. It exposes
stable, typed mathematical operations that agents can discover and compose. It
does not attempt to enumerate all mathematical knowledge, mirror every backend
API, or encode proof strategies.

The caller owns problem representation, decomposition, sequencing, strategy,
interpretation, and stopping. Jacobian owns the public mathematical contracts,
typed values, execution bounds, and honest result semantics around each move.

## A dependency graph of executable results

Mathematics is built in layers: foundational theories support mathematical
objects, and those objects support lemmas and larger results. Jacobian does not
try to encode that entire hierarchy or turn every lemma into a separate tool.
It exposes selected reusable results as typed operations that agents can
combine:

```text
mathematical objects -> executable operations -> larger investigations
                              ^       |
                              |       v
                        typed results
```

The operation vocabulary is therefore closer to a dependency graph than to a
flat list or a simple tree. One operation can use several mathematical inputs,
and its result can be reused by many later operations. “Atomic” describes the
boundary of one operation: it establishes one clear, independently useful
mathematical postcondition. It does not mean that the operation is mathematically
indivisible or internally simple.

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

### Locate the reusable boundary inside a proof

A specialized proof often contains useful mathematics at several scales. The
right operation is neither automatically the smallest line of code nor the
whole theorem workflow.

| Proof fragment | Disposition | Boundary lesson |
| --- | --- | --- |
| One floating nullspace step | Too low-level | It exposes an algorithmic move without the preservation and error guarantee callers need. |
| Hard-constraint rounding | Atomic candidate | Preserving specified linear equations while making bounded progress on fractional coordinates is a stable postcondition; the entire allocation proof remains caller reasoning. |
| Total variation of materialized finite tables | Atomic candidate | The source and linear comparison work are explicit. Accepting succinct product distributions is a materially broader contract because their support expansion can be exponential and the compact problem can have different complexity. |
| Weak-design construction | Atomic candidate | A bounded design with stated overlap parameters is reusable; extractor reconstruction and the entropy contradiction are the global proof lift. |
| Factorial insertion for an arbitrary callback | Reasoning | The identity belongs in a theorem or caller derivation rather than an operation whose behavior is defined by executable caller code. |

State the source representation as part of this boundary. Materialized,
succinct, generated, and oracle-backed inputs can denote related objects while
requiring different admission proofs, algorithms, and result bounds.

### The carrier ladder for smooth and analytic mathematics

Mathematical data that sounds geometrically related need not belong to one
interchangeable record.  Treat the following as a ladder of increasingly rich
semantics, rather than as fields to add opportunistically to a generic
`geometry` value:

```text
exact scalars
  -> algebraic or complex values
  -> coordinate-local tensors on a specified chart
  -> compatible chart atlases
  -> global smooth or analytic objects
  -> PDE evolution, approximation, convergence, and limit semantics
```

The lower layers can support useful atomic operations.  For example, a
gradient of a rational function, a coordinate curvature profile of a
nondegenerate rational metric, or a coordinate Lie derivative can each have a
precise input, output, and defining identity.  They do not thereby define a
smooth manifold, a global Riemannian geometry, or a Ricci flow.

Moving up the ladder requires the data and laws of the next layer: transition
maps and compatibility for an atlas; regularity, bundles, and global
topological compatibility for smooth objects; and a time-dependent state,
norm/error model, domain, and convergence semantics for analytic estimates or
PDEs.  A finite mesh, samples, or a rational local chart can be an honest input
to a different bounded operation, but must not be presented as an exact solver
for a global or limiting argument simply because a source proof used related
vocabulary.

### Techniques, functions, and operations

An established technique is evidence for executable vocabulary, not an
automatic public operation. Classify the mathematical result it establishes
before deciding how Jacobian should expose it:

| Technique role | Jacobian disposition |
| --- | --- |
| Computes an independently useful bounded result | Public-operation candidate |
| Useful exact helper without sufficient catalog leverage | Native-only function |
| Alternative algorithm for an existing result | Private kernel |
| Defines or reconstructs an operation's result | Invariant and fixture |
| Organizes several moves into a proof | Caller reasoning |

Make established terminology searchable through the relevant public operation,
even when the technique remains its private kernel. Admit a separate operation
only when the technique returns a distinct reusable result, witness,
decomposition, or certificate. Multiple algorithms for the same mathematical
input, output, and defining relation belong behind one canonical operation.

For example, suppose `ramanujan_sum.compute(q, n)` is admitted to return the
exact integer $c_q(n)$. “Divisor–Möbius formula” should be searchable
terminology for that operation. A general Möbius transform may be separately
runnable because it has a different postcondition. A duplicate
`ramanujan_sum_via_mobius.compute` should not exist when it merely returns the
same scalar through another algorithm.

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

Whether the agent invoked Jacobian is not a prerequisite for gap discovery.
Bespoke mathematical code can reveal a missing reusable postcondition;
availability, discovery, selection, and execution are separate adoption
questions.

Treat the trajectory as evidence, not as the operation specification. Bind an
availability claim to the catalog and repository revision visible in that
session, and distinguish an operation that was unavailable, undiscovered,
unselected, called incorrectly, or genuinely insufficient. Preserve the exact
scope of the fallback result as well: a numerical candidate, finite search, or
successful one-off computation does not establish an exact general contract.
Verify the current catalog and source before turning the observation into an
operation proposal.

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

Gap diagnosis and public-operation admission are separate decisions. A focused
issue may record a reusable missing postcondition even when the available
evidence does not yet establish enough leverage for the agent-visible catalog,
or when review may keep it native-only, split its postconditions, or remove it.
Admission gates decide that later disposition; they should not suppress an
evidence-backed gap record. Conversely, recording an operation gap does not
pre-admit a catalog operation.

### Common boundary mistakes

| Candidate description | Usual disposition | Why |
| --- | --- | --- |
| “Return the next proof step.” | Reasoning or workflow | The result depends on strategy, available lemmas, and a caller-selected stopping rule. |
| “Solve this named conjecture.” | Theorem wrapper | It bundles representation, search, interpretation, and conclusion around one motivating statement. |
| “Run the backend's algorithm X.” | Private kernel or discovery vocabulary | An algorithm is not a public result; retain it privately unless it establishes a distinct postcondition. |
| “Compute the canonical source-indexed profile of this bounded family.” | Plausible operation candidate | A complete, independently useful relation can compose with multiple later arguments when its carrier and bounds are explicit. |
| “Add a new operation because the existing one rejects large `n`.” | Scale/backend or contract review first | The existing postcondition may be right; derive work and output bounds before duplicating it. |
| “Discretise a continuous source argument and expose the same name.” | Representation prerequisite, reasoning, or no action | Samples or meshes do not automatically retain the source's metric, error, regularity, or limit semantics. |

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
