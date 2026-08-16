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
the operation and adapt it when a request contract changes. The catalog test
checks every published example at the request boundary; the operation's owning
tests should exercise any nontrivial example behavior.

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
