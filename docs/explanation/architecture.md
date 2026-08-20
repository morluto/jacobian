# Architecture

The [product blueprint](product-blueprint.md) owns Jacobian's product model.
This page describes the package boundaries and ordinary execution path that
implement it.

The serving process compiles one immutable catalog directly from explicit
`MathTool` entries and exposes `math.find` and `math.run` through the MCP Python
SDK.

Each operation validates one typed request, calls a domain-owned mathematical
function or maintained private backend, and returns one typed bounded result.

The ordinary call path is:

```text
operation ID + JSON -> declaration -> Pydantic request -> domain function -> typed result
```

The domain function may compose a maintained backend such as SymPy, FLINT,
NetworkX, or Z3 where that algorithm is relevant. Those backends remain private
computational engines behind Jacobian's public mathematical contracts.

Domain values live beside the functions that own their semantics under
`jacobian.math.<domain>`. HNF, LLL, and Smith-related direct computations call
maintained backends in process; a subprocess is retained only where actual
external isolation is required.

Each mathematical owner keeps its public values and functions in ordinary
semantic modules, private Pydantic wire models in `_models.py` where needed,
and its immutable `TOOLS` tuple in `_tools.py`. Its `_admission.py` binds those
tools and their decisions into one owner-local `REGISTRATION`. Catalog
construction discovers only packaged `_admission.py` modules under
`jacobian.math`, sorts their module paths, validates every registration, and
then freezes the resulting built-in inventory. There is no central domain list
and no external plugin discovery. `jacobian.catalog` owns declaration models,
search, and immutable lookup; `jacobian.dispatch` owns strict invocation;
`jacobian.mcp` and the CLI are delivery boundaries. The private root model and
exact-scalar helpers contain only behavior genuinely shared by unrelated
owners.

## Package organization and family folding

A domain is a top-level `jacobian.math.<family>` package when it owns a
distinct canonical value type and imports no other family's `values`. A domain
that consumes a family's canonical value type is a subpackage of that family,
not a top-level package. This keeps the top level free of ticket-shaped feature
packages while each capability keeps its own values, models, backends, and
tests.

Decide by evidence, in this order:

1. Shared value type. A domain that imports a family's `values` module (for
   example `matrices.values.RationalMatrix`) belongs to that family.
2. Operation-ID domain prefix. The first segment of an operation ID
   (`graph.*`, `matrix.*`, `polynomial.*`, `formal_series.*`) names the
   mathematical family even when the package name does not. The prefix is a
   discovery value: never rename operation IDs to follow a package move.
3. Self-containment. A domain with its own value type and no import of another
   family's `values` remains top-level (for example `formal_power_series`,
   `root_isolation`, `electrical_networks`).

Nest into a subpackage when the capability has its own
values/models/operations/tools/tests, and into a module when it is a lone
native capability. Drop a now-redundant family prefix when nesting
(`matrix_analysis` -> `matrices/analysis`, `graph_coloring_ops` ->
`graphs/coloring`), and keep descriptive names otherwise.

A fold preserves operation IDs and request/result schemas, keeps one math
owner per tool (request, result, and run share the first path segment), deletes
the old path in the same change, and lands as one family per change.

Logic follows the same rule. CNF canonicalization and assignment checks are
pure direct operations. SAT and bounded QF SMT-LIB solving call the maintained
Z3 Python binding in process. `lean.check` is a one-shot external boundary: it
writes one source file in a request-scoped temporary directory, invokes the
fixed Lean environment with an explicit timeout, returns typed diagnostics, and
deletes that directory.

Remote serving uses the same immutable operation library. Authentication
produces a small request-scoped context. Deployment supplies an immutable
service artifact, configuration, and health checks; rollout, rollback, and
persistence remain deployment-platform responsibilities.
