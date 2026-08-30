# Product blueprint

Jacobian exposes stateless functions over typed mathematical values through two
MCP verbs: find an operation and run one operation. The caller owns composition:
decomposition, sequencing, retention of values, and stopping.

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
