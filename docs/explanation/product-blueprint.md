# Product blueprint

Jacobian is a native mathematical library with an immutable catalog of published
operations. MCP exposes that catalog through two verbs: find an operation and
run one operation. Native callers can use domain functions without a catalog or
server. The caller owns composition: decomposition, sequencing, retention of
values, and stopping.

Mathematical owners define values, operation semantics, admission, and typed
outcomes. Publication makes selected capabilities discoverable; it does not
create a second mathematical implementation. The server owns protocol delivery,
authorization, and deployment policy around those library contracts.

The operation path and ownership boundaries are defined in the
[architecture](architecture.md). A domain kernel may use a maintained library
such as SymPy, FLINT, NetworkX, or Z3 privately; Jacobian owns the public
mathematical semantics, bounds, typed outcomes, and canonical boundary around
that computation.

Built-in operations are explicit immutable declarations. Ordinary results are
small bounded mathematical values returned directly. Direct mathematical
predicates, such as checking a SAT assignment, own their request and result
alongside the mathematics.
