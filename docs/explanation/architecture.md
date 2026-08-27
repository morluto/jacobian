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

## Transport and mathematical ownership

The MCP Python SDK owns the fixed transport boundary: registration of
`math.find` and `math.run`, their outer argument and output schemas, protocol
validation, and structured JSON delivery. Jacobian does not duplicate those
checks.

`math.run` still needs a small dispatch boundary because its `payload` has an
operation-specific schema that is known only after its immutable `operation_id`
is resolved. Dispatch therefore does only this: resolve the declaration, parse
the payload once with that owner's request model, invoke that owner once, and
project the typed result once. It does not contain domain admission, backend
logic, result-specific replay, or workflow state.

Jacobian is a typed, bounded tool layer over maintained mathematical libraries.
For each accepted `math.run` request, the owner computes one semantic admission
plan, executes one bounded kernel, and constructs the canonical result from
that trusted work. Request and result models perform only structural
validation; they do not rerun admission, a backend, enumeration, or a defining
relation.

A replay is permitted only through a named, bounded owner-local verifier for
independently supplied theorem-bearing data. It is an explicit trust-boundary
operation, never a side effect of ordinary execution or deserialization.

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

An immutable declaration may carry a small `discovery_terms` vocabulary of
reviewed, established names for its exact postcondition. Terms are catalog
metadata used by deterministic `math.find` ranking; they do not alter the
canonical title and description, operation ID, request syntax, or mathematical
claim. This keeps ordinary morphology in the shared lexical normalizer and
domain terminology with the owner that can review its meaning.

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

## Bounded worker adapters

Use a child process only for a concrete isolation, killability, or fixed-toolchain
need. The mathematical owner remains responsible for the complete request
envelope: it admits the request, retains its canonical source, starts the
worker, and constructs the final result. The worker receives one strict payload
and returns only a bounded derived projection. The parent binds that projection
to its admitted source before result validation; a worker does not echo or
replace retained canonical values.

The owner charges parsing, launch, backend work, projection, validation, and
cleanup against one local execution plan and deadline. Worker capture limits
cover the actual UTF-8 projection, not an assumed public-result shape. A new
process owner must be named in both the architecture check and the import
contract, with a concrete backend rationale. The detailed codec, cleanup, and
typed-failure rules live in the [mathematical backend contract](../reference/mathematical-backends.md#child-process-adapters).

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
pure direct operations. SAT and bounded QF SMT-LIB solving use the maintained
Z3 Python binding through bounded owner-local workers.

Remote serving uses the same immutable operation library. Authentication
produces a small request-scoped context. Deployment supplies an immutable
service artifact, configuration, and health checks; rollout, rollback, and
persistence remain deployment-platform responsibilities.
