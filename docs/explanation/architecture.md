# Architecture

Jacobian is a stateless mathematical kernel. Its serving process loads one
immutable package index of explicit `InlineOperation` declarations and exposes
`math.find` and `math.run` directly through the MCP Python SDK.

Each operation validates one typed request, calls a domain-owned mathematical
function or maintained private backend, and returns one typed bounded result.
The kernel has no SQLite database, catalog overlay, registry, schema registry,
artifact repository, value-reference store, workspace, selected-family router,
or generic verification service.

Domain values live beside the functions that own their semantics under
`jacobian.math.<domain>`. Backends such as SymPy, NetworkX, and Python-FLINT
are private implementation details. HNF, LLL, and Smith-related direct
computations call maintained backends in process; a subprocess is retained only
where actual external isolation is required.

Logic follows the same rule. CNF canonicalization and assignment checks are
pure inline operations. SAT and bounded QF SMT-LIB solving call the maintained
Z3 Python binding in process. `lean.check` is a one-shot external boundary: it
writes one source file in a request-scoped temporary directory, invokes the
fixed Lean environment with an explicit timeout, returns typed diagnostics, and
deletes that directory. There are no Lean sessions, declaration caches, replay
records, or proof-state resources.

Remote serving shares the immutable operation library. Authentication produces
a small request-scoped context; it does not create a tenant runtime or tenant
state. Deployment supplies an immutable service artifact, configuration, and
health checks. Rollout, rollback, and persistence belong to deployment
infrastructure, not Jacobian.
