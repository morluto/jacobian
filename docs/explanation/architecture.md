# Architecture

Jacobian is a stateless mathematical kernel. Its serving process compiles one
immutable catalog directly from explicit `MathTool` entries and
exposes `math.find` and `math.run` through the MCP Python SDK.

Each operation validates one typed request, calls a domain-owned mathematical
function or maintained private backend, and returns one typed bounded result.
The kernel has no SQLite database, catalog overlay, registry, schema registry,
artifact repository, value-reference store, workspace, selected-family router,
or generic verification service.

The ordinary call path has no intermediate lifecycle product:

```text
operation ID + JSON -> declaration -> Pydantic request -> domain function -> typed result
```

The domain function may compose a maintained backend such as SymPy, FLINT,
NetworkX, or Z3 where that algorithm is relevant. Those backends remain
private computational engines; they do not add a provider runtime, worker
protocol, or competing public type system.

Domain values live beside the functions that own their semantics under
`jacobian.math.<domain>`. Backends such as SymPy, NetworkX, and Python-FLINT
are private implementation details. HNF, LLL, and Smith-related direct
computations call maintained backends in process; a subprocess is retained only
where actual external isolation is required.

Each mathematical owner keeps its public values and functions in ordinary
semantic modules, private Pydantic wire models in `_models.py` where needed,
and its immutable `TOOLS` tuple in `_tools.py`. There is no parallel
`contracts/` or `domains/` package. `jacobian.catalog` owns declaration models,
explicit built-in imports, search, and immutable lookup; `jacobian.dispatch`
owns strict invocation; `jacobian.mcp` and the CLI are delivery boundaries.
The private root model and exact-scalar helpers contain only behavior genuinely
shared by unrelated owners.

Logic follows the same rule. CNF canonicalization and assignment checks are
pure direct operations. SAT and bounded QF SMT-LIB solving call the maintained
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
