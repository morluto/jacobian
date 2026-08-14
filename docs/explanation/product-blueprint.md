# Product blueprint

Jacobian is an MCP server for two atomic mathematical verbs: find an operation
and run one operation. It is not a workflow engine, workspace, artifact store,
or mathematical project manager.

The server owns typed operation contracts, strict request validation, resource
bounds, immutable discovery, and the final MCP projection. The caller owns
decomposition, sequencing, persistence of any values it needs, and stopping.

For an ordinary call, Jacobian selects one immutable declaration, parses its
Pydantic request once, calls the domain-owned function, and returns that
function's concrete typed result. The function may use a maintained library
such as SymPy, FLINT, NetworkX, or Z3 privately; Jacobian owns the public
mathematical semantics, not a second provider/runtime abstraction around that
library.

Built-in operations are explicit immutable declarations. Ordinary results are
small bounded values returned directly. No operation can silently publish a
durable artifact or retain caller state. A direct mathematical predicate—such
as checking a SAT assignment—owns its request and result alongside the
mathematics. The kernel has no generic checker registry, receipt, or record
service.
