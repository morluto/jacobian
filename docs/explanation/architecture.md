# Architecture

The [product blueprint](product-blueprint.md) owns Jacobian's product model.
This page describes the package boundaries and ordinary execution path that
implement it.

The serving process compiles one immutable catalog directly from explicit
`MathTool` entries and exposes `math.find` and `math.run` through the MCP Python
SDK.

Each operation parses one typed request, performs owner-local request admission,
executes one bounded Jacobian kernel path (which may use a maintained private
backend), constructs one canonical typed result, and returns it through one
transport projection.

The ordinary call path is:

```text
operation ID + JSON
  -> declaration
  -> strict typed request
  -> owner-local request admission and execution plan
  -> bounded Jacobian kernel or private backend adapter
  -> canonical typed result construction
  -> MCP/JSON transport projection
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

Catalog admission decides publication and is not runtime planning. The
mathematical owner decides request admission, builds the request-scoped
execution plan, owns the backend adapter, and constructs the canonical result.
Defining-invariant evidence belongs in the operation's tests; a full replay is
not part of ordinary execution. An adapter may reject malformed backend data
while converting it, but that is integration safety rather than a separate
mathematical result stage.
Dispatch and MCP project an already-admitted typed result into the final
transport envelope; they must not discover a mathematical or work bound only
after execution. Independently supplied result data uses an explicit, bounded
replay verifier rather than ordinary result construction.

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
Z3 Python binding in process.

Remote serving uses the same immutable operation library. Authentication
produces a small request-scoped context. Deployment supplies an immutable
service artifact, configuration, and health checks; rollout, rollback, and
persistence remain deployment-platform responsibilities.
