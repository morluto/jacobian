# Domain operation library

Every built-in operation is a direct typed mathematical function with one
domain owner. Declaration modules export immutable tuples of
`MathTool` values. `math.find` reads those entries and `math.run`
validates then executes exactly one of them.

The ordinary path is: select declaration, parse its Pydantic request once,
call the domain function, and return its concrete result. A domain function may
use a maintained library privately for its algorithm; callers see Jacobian's
typed mathematical values, not backend objects.

Keep values, codecs, invariants, and backend conversions with their domain.
Shared contracts are limited to passive cross-domain primitives. A bounded
operation reports mathematical completeness or uncertainty in its own result.

Public request contracts must make their valid representation visible before a
backend call. Express constraints that JSON Schema can represent in typed field
metadata. When a domain invariant needs a Pydantic model validator—such as a
cross-field relation or canonical term ordering—also provide an explicit field
or model description and a minimal valid example in the exported schema. The
validator remains authoritative; the metadata lets a caller form a valid first
request rather than discover the rule only through a rejected call.

Every built-in `MathTool` declaration must publish at least one small valid
invocation example. An example is part of the public contract: it must validate
against the declaration's request model, use canonical values where required,
and be executable in Jacobian's supported local environment. Keep it close to
the operation and adapt it when a request contract changes. The composition
catalog test executes every published example: the payload must validate, the
domain function must return a typed result, and that result must re-validate.
The operation's owning tests still own nontrivial example behavior and the
adversarial and request-boundary cases in the preflight below. Those two cases
are written with the operation; they are not a separate CI program.

Write an invocation example's description in two parts: first state the
computation the operation performs on the supplied values, then state the
important precondition that makes the example valid. For example, use
`Compute the exact eigenvalues of [[1, 2], [3, 4]]; the matrix must be square
and rectangular.` The first part tells an agent what the operation does; the
second part teaches the input rule it must preserve. A precondition by itself,
such as `The matrix must be square`, is not an adequate example description.

Examples help an agent form its first request without guessing. JSON Schema can
name fields and simple bounds, but it cannot fully communicate validator-owned
rules such as nested value shape, canonical ordering, coupled fields, or the
smallest useful composition. On exact inspection, an agent can copy an example
payload and adapt its mathematical content instead of discovering that wire
contract through a failed call, a lengthy ad-hoc script, or trial and error.
Examples illustrate a valid representation; they do not prescribe a proof
strategy or restrict how operations may be composed.

Avoid **validator-only public contracts**. Do not introduce a required input
representation solely through a Pydantic validator and expect callers to infer
it from an error. When a rule cannot be expressed as an ordinary JSON Schema
constraint, pair the validator with schema-visible field or model guidance and
a valid invocation example. Diagnostics remain the recovery path for malformed
requests; they are not the primary documentation for an operation's wire
contract.

Use maintained backends through thin private adapters. Direct bounded results
compose by being supplied as the next operation's typed payload.

The logic family illustrates the boundary. `sat.cnf.canonicalize` returns a
canonical CNF value; `sat.assignment.check` and `sat.solve` accept that value
directly. `smt.solve` accepts one bounded QF SMT-LIB query. `lean.check` accepts
one bounded source snippet and returns elaboration diagnostics after a one-shot
process invocation.

## Operation preflight

First diagnose the gap: a missing operation is only one of several possible
responses to an observed composition failure. Classify the failure as
representation, interoperability, discovery, contract, scale/backend, operation,
or reasoning before designing an implementation (see
[Executable mathematical vocabulary](../explanation/executable-mathematical-vocabulary.md)).
Only a genuine operation gap proceeds to the
[admission contract](public-operation-admission.md).

Every new or materially changed public operation must include the following
completed review artifact in its issue or pull request. A field may say `Not
applicable` with a reason; it must not be omitted.

### Public operation contract

- Mathematical input domain:
- Canonical public value type:
- Producer/consumer closure, or why not applicable:
- Degenerate inputs:
- Parent/ring/field identity:
- Deterministic work bound:
- Backend and supported version:
- Backend input domain:
- Conversion/coercion behavior:
- Result type:
- Reconstruction or defining invariant:
- Typed execution failures:
- Property and boundary tests:

These checks have distinct owners. Admission validation proves that a request
belongs to the advertised domain. Backend conversion converts an already valid
value; it does not widen or discover that domain. Backend result validation
checks integration and reconstruction. Result validation must never compensate
for an overbroad request contract.

Do not add a public operation until its stated mathematical claim has a bounded,
appropriate implementation. A public operation is the `MathTool` contract—its
identifier, typed request and result, scope, and mathematical claim—not merely
a native Jacobian function or maintained backend routine. It may adapt either,
but its claim must be no broader than the implementation can establish.

A heuristic or approximation may be useful only when its result contract states
that limited scope. It must not return a negative decision, exact invariant, or
optimum that the implementation cannot establish.

Verify that the adapter preserves the claimed semantics. Do not present a
heuristic, approximation, or solver `UNKNOWN` as an exact conclusion; coerce
exact values to floating point; confuse similarly named invariants; or discard
backend information the result contract needs, such as multiplicities, bases,
or witnesses.

Before declaring the operation, provide tests for:

- a known-answer input and its claimed mathematical result;
- a boundary or degenerate input, including valid empty, zero, singleton, or
  identity values where the domain admits them;
- an adversarial input that distinguishes the stated semantics from a tempting
  weaker algorithm;
- a public-operation assertion that the returned value satisfies its defining
  mathematical invariant or witness, rather than merely parsing or reaching a
  backend; and
- request validation proving a schema-valid input either returns a typed
  result or is rejected by the request model—never a host exception.

Apply these adapter and request-boundary rules:

- Every public string field that carries mathematical syntax must document a
  finite grammar and have a test proving that parsing does not evaluate caller
  text. Do not pass caller input to `sympify`, `parse_expr`, `eval`, `exec`, or
  an evaluator generated by `lambdify`. Prefer canonical term or AST values as
  the authoritative contract; a textual convenience parser, if one exists,
  must construct the same value from an explicit allowlist.
- For every backend routine, record the coefficient domain, dimensional or
  degree limits, structural preconditions, degenerate cases, and resource
  limits it accepts. Encode those constraints in the concrete request model so
  an accepted request does not discover the backend domain through an
  exception.
- Every exact decomposition, certificate, or authoritative derived value must
  state its defining reconstruction or preservation equation and test it. Do
  not infer a mathematical property from the shape of lossy backend output or
  discard units, multiplicities, generators, axes, quotient maps, or other data
  needed to reconstruct or compose the result.
- Canonical integer and rational strings must reach backends only through
  `parse_canonical_integer()`, `as_integer_ratio()`, or an owner conversion
  helper. When the contract permits values above CPython's 4,300-digit integer
  string conversion limit, every adapter must include a test above that limit.
- For operations with mathematical preconditions such as nonsingularity,
  uniqueness, irreducibility, or nondegeneracy, tests must cover each excluded
  class and prove rejection occurs during request validation.

Before publication, record one owner-local admission decision in the
mathematical domain's `_admission.py` module. `jacobian.catalog.admission` owns
the shared policy types and fail-closed validation (see the
[public operation admission](public-operation-admission.md) contract).

### Domains, parents, and coercion

Canonical values carry the context needed to determine their mathematical
meaning. A polynomial includes its coefficient domain, generators, and
ordering where relevant. An ideal belongs to exactly one polynomial ring. A
finite-field element includes its field presentation. Matrices and
authoritative derived tables retain their axes and parent domain.

The same serialized expression in two contexts need not denote the same value.
Require exact parent identity for ordinary operations. A deliberate change of
ring, field, parent, generators, or axes must use a named operation or typed
morphism whose behavior is part of its contract. Never silently map unmatched
variables to zero. Backend generator inference, ambient rings, and automatic
coercion are private conveniences and do not define Jacobian's public
semantics.

Each mathematical value has one canonical type owned by its domain. A producer
returns that type and downstream consumers accept it unchanged. An operation
request may contain the value alongside genuine operation parameters, but must
not redefine the value as a parallel collection of fields. Callers must not
have to remember and reattach a field, ordered axis, ranked signature, ambient
dimension, or other mathematical context. This closure rule applies to empty
and degenerate values too: for example, a zero-row matrix still retains its
declared column axis.

When a producer-consumer relationship exists, the operation review artifact
must name it and its tests must pass the producer's serialized value directly
through the consumer's typed boundary. Do not introduce a generic value
registry or universal mathematical-object base class for this purpose; reuse
the owner domain's concrete value type.

Classify public outputs before choosing their schema:

| Output kind | Contract |
| --- | --- |
| Canonical value | A complete reusable mathematical object accepted by its downstream consumers. |
| Source-bound result | A source value plus a conclusion or certificate whose defining relation is validated. |
| Display projection | A human-readable summary that is not accepted as a composable mathematical value. |

For every producer or materially changed consumer, answer all of the following
in the producer/consumer closure field of the review artifact:

- What domain-owned canonical type does the producer return?
- Which downstream operations consume that type?
- Can its serialized value be supplied to each consumer unchanged?
- Does it retain its parent, presentation, ordered axes, ambient dimension, and
  normalization where those determine its meaning?
- What mathematical context remains present for empty, zero, identity, or
  otherwise degenerate values?
- Is each decision or certificate bound to the source value it concerns?
- Can result validation replay the defining relation within the declared work
  bound?

Decision and profile results are relations, not detached booleans or numbers.
Retain the source values needed to state the relation and replay its defining
equation in result validation. A compact result may omit a large derivation
ledger when bounded replay from the retained source is deterministic, but it
must not accept an authored conclusion merely because its scalar fields have
the right shape.

Backend integration follows the reusable
[mathematical backend contract](mathematical-backends.md).

### Static and executable enforcement

Keep static policy limited to boundaries that syntax can identify reliably.
The architecture checker forbids evaluator-capable parsing in the mathematical
tree and confines process execution to explicit owners behind the shared
supervisor. The fail-closed admission ledger and catalog conformance tests—not
an approximation in the linter—prove that public declarations use the standard
validation and execution path and do not expose backend values.

Mathematical correctness, parent compatibility, reconstruction, and backend
domain support require executable contract and property tests. Do not encode
those semantic claims as source-text or private-helper lint rules.

### Boundedness proof

Jacobian's operations are reusable mathematical instruments for agents doing
high-level mathematics and investigating conjectures. Treat boundedness as
part of the mathematical contract, not as a property of the transport or a
final serializer. For each operation, write down three
different obligations:

1. **Input domain:** which mathematical objects and degenerate cases are
   accepted, and which are excluded as inapplicable?
2. **Computation:** what bounds the algorithm's work and intermediate values
   before the backend expands, enumerates, or solves anything?
3. **Output:** what bounds the exact returned value, witness, residual, or
   certificate, and how is that bound related to the accepted input domain?

The request contract must enforce the first obligation and the preconditions
needed for the second and third. A backend or result conversion may still
validate an invariant, but it must not be the first place an accepted request
discovers that its exact answer is too large. If a bound is conservative, name
the quantity it bounds, state why it is safe for the algorithm, and test both
the rejected adversarial case and a useful case near the boundary. Do not use a
post-hoc output-term cap, truncation, sentinel, or host exception as a hidden
computational budget.

When an operation has a genuine incomplete or unknown outcome, expose that
state in its domain result with the evidence and bounds needed to interpret it.
Do not turn an inability to finish or represent the exact answer into a
mathematical conclusion. When no such result is defined, narrow the request
domain until every accepted request returns the declared typed value.

CI executes every advertised invocation example and a bounded deterministic
mutation set derived from those examples. For every mutation accepted by the
concrete request model, the operation must return its declared result type and
must not leak a host or backend exception. The adversarial semantic case and
the schema-valid request-boundary case still belong in the owning domain tests;
generic mutations can expose admission gaps but cannot prove domain-specific
mathematical correctness.

If no bounded implementation can support the public claim, do not expose the
operation yet. A backend import or native function is not evidence that its
result has the desired mathematical semantics.
