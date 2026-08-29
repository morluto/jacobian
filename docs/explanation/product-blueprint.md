# Product blueprint

Jacobian exposes each admitted stateless function over typed mathematical
values as a directly callable MCP tool. Its operation ID is also its MCP tool
name, and its owner-local request and result models are the callable schemas.
`math.find` separately provides semantic mathematical catalog search and exact
contract inspection. The generic `math.run` path remains temporarily available
while direct discovery is evaluated; ordinary direct calls do not require it.
The caller owns composition: decomposition, sequencing, retention of values,
and stopping.

The server owns typed operation contracts, strict request validation, resource
bounds, immutable discovery, and the final MCP projection.

The operation path and ownership boundaries are defined in the
[architecture](architecture.md). A domain kernel may use a maintained library
such as SymPy, FLINT, NetworkX, or Z3 privately; Jacobian owns the public
mathematical semantics, bounds, typed outcomes, and canonical boundary around
that computation.

Built-in operations are explicit immutable declarations. Ordinary results are
small bounded mathematical values returned directly. Direct mathematical
predicates, such as checking a SAT assignment, own their request and result
alongside the mathematics.
