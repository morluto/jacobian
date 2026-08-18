# Domain operation library

Every built-in operation is a direct typed mathematical function with one
domain owner. Declaration modules export immutable tuples of
`MathTool` values. `math.find` reads those entries and `math.run`
validates then executes exactly one of them.

The ordinary path is: select declaration, parse its Pydantic request once,
call the domain function, and return its concrete result. A domain function may
use a maintained library privately for its algorithm; callers see Jacobian's
typed mathematical values, not backend/provider objects.

Keep values, codecs, invariants, and backend conversions with their domain.
Shared contracts are limited to passive cross-domain primitives. A bounded
operation reports mathematical completeness or uncertainty in its own result;
it does not add a generic assurance, artifact, publication, replay, or
verification wrapper.

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
compose by being supplied as the next operation's typed
payload.

The logic family illustrates the boundary. `sat.cnf.canonicalize` returns a
canonical CNF value; `sat.assignment.check` and `sat.solve` accept that value
directly. `smt.solve` accepts one bounded QF SMT-LIB query. `lean.check` accepts
one bounded source snippet and returns elaboration diagnostics after a one-shot
process invocation. None of these operations consumes or produces a stored
reference.

## Operation preflight

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

Before publication, record one admission decision for the operation in
`src/jacobian/catalog/admission.py`; catalog construction fails closed without
it (see the
[public operation admission](public-operation-admission.md) contract).

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

CI executes every advertised invocation example. The adversarial case and the
schema-valid request-boundary case belong in the owning domain tests.

If no bounded implementation can support the public claim, do not expose the
operation yet. A backend import or native function is not evidence that its
result has the desired mathematical semantics.
