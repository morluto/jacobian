# Product blueprint

Jacobian is an MCP server for two atomic mathematical verbs: find an operation
and run one operation. It is not a workflow engine, workspace, artifact store,
or mathematical project manager.

The server owns typed operation contracts, strict request validation, resource
bounds, immutable discovery, and the final MCP projection. The caller owns
decomposition, sequencing, persistence of any values it needs, and stopping.

Built-in operations are explicit immutable declarations. Ordinary results are
small bounded values returned inline. No operation can silently publish a
durable artifact or retain caller state. A direct mathematical predicate—such
as checking a SAT assignment—owns its request and result alongside the
mathematics. The kernel has no generic checker registry, receipt, or record
service.
