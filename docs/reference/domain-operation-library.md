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

Use maintained backends through thin private adapters. Direct bounded results
compose by being supplied as the next operation's typed
payload.

The logic family illustrates the boundary. `sat.cnf.canonicalize` returns a
canonical CNF value; `sat.assignment.check` and `sat.solve` accept that value
directly. `smt.solve` accepts one bounded QF SMT-LIB query. `lean.check` accepts
one bounded source snippet and returns elaboration diagnostics after a one-shot
process invocation. None of these operations consumes or produces a stored
reference.
