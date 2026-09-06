# Architecture

The [product blueprint](product-blueprint.md) owns Jacobian's product model.
This page describes the package boundaries and ordinary execution path that
implement it.

The serving process compiles one immutable catalog directly from explicit
`MathTool` entries and exposes `math.find` and `math.run` through the MCP Python
SDK. Matching and exact inspection are the two request modes of `math.find`.

For most native functions, the ordinary call path is deliberately small:

```text
user calls a function
  -> validate and normalize its mathematical input
  -> perform the computation
  -> return a canonical Jacobian value
```

Request admission is part of validation, not a requirement to build an
orchestration layer. A simple operation may use one owner-local guard that
returns nothing on success, followed directly by its kernel. An operation needs
a distinct execution-plan value only when admission derives information that
execution genuinely reuses, such as an algorithm choice, prepared finite data,
or work and output reservations.

Wire-model validation is narrower than native operation admission. Pydantic may
enforce field types, canonical encodings, cheap representation limits, and
intrinsic cross-field relations needed to construct the request value. It must
not call a mathematical backend, enumerate a search space, select an algorithm,
reserve result work, or cache execution data in a private model attribute.

The MCP path wraps that same native operation with discovery, wire parsing, and
transport projection:

```text
operation ID + JSON
  -> declaration
  -> strict typed request
  -> owner-local native operation
  -> bounded Jacobian kernel or private backend adapter
  -> canonical typed result construction
  -> MCP/JSON transport projection
```

### Runtime ownership rule

Parsing establishes only the canonical request shape and cheap representation
limits. The owner then makes one semantic admission decision for the
invocation. When that decision produces facts the kernel or trusted result
construction reuses—the selected algorithm or backend, derived work and output
bounds, or prepared finite data—the owner may represent them as a request-scoped
execution plan. Otherwise, a guard followed directly by the kernel is the
preferred shape. "One semantic admission decision" means one consistent check
and any reusable derived facts; it does not require a plan class, planner
abstraction, or extra module for every operation.

Request and ordinary result models must not perform semantic admission, call a
backend, enumerate candidates, or check a defining relation. A wrapper or
kernel must not recompute an admission quantity already established by
admission or held by a plan.

## Design and review principles

These are defaults for keeping an operation small, trustworthy, and easy to
change. They are ownership rules, not a framework that every owner must
instantiate.

- **Design by contract.** State the request representation, mathematical
  postcondition, result states, defining invariant, and resource envelope
  before choosing an implementation. A generated schema is part of the
  contract: it must not advertise requests that runtime admission rejects.
- **One owner and one source of truth.** Put each mathematical policy and
  derived bound with the owner that can enforce it. A shared canonical value
  should describe mathematical meaning, not one operation's incidental work
  limit. Put operation-specific limits in operation admission; introduce a
  subtype only when its mathematical meaning or invariants differ for its
  consumers, not merely because one operation has a different ceiling.
- **Functional core, imperative boundary.** Keep canonical values and pure
  mathematical transformations separate from backend calls, process control,
  deadlines, filesystem access, and MCP delivery. The imperative boundary
  owns those effects and translates them into the domain's typed outcomes.
  This is a ports-and-adapters boundary: native APIs, maintained backends,
  child workers, and MCP delivery adapt to the owner-local contract rather
  than defining that contract independently.
- **Fail fast, then compute once.** Reject cheap malformed or over-budget
  representations before expensive work. After canonicalization, compute
  semantic admission once and reuse its facts; do not repeat a size probe or
  admission calculation in a request model, wrapper, worker, and result
  constructor.
- **Make illegal states unrepresentable.** Use discriminated result states,
  source-bound context, and schema-visible bounds when callers must distinguish
  cases. Defense in depth belongs at trust boundaries; it is not a reason to
  scatter conflicting copies of the same policy through every layer.
- **Adversarial closure.** A change is not closed when only its happy path
  works. Test accepted near-boundary inputs, malformed and deeply nested
  boundary data, schema/runtime parity, native/MCP parity, downstream consumers,
  and every mandatory deadline or serialization phase.
- **Smallest useful abstraction.** Start with explicit owner-local code. Extract
  a shared helper only after multiple owners have the same mechanics and
  contract. A worker projection may be reused as an internal IPC protocol only
  when its framing, version, bounds, source binding, and failure semantics are
  explicit; it must not become an accidental public result format.

Ordinary execution does not replay its own computation. Defining-invariant
checks normally belong in the owning tests. When checking caller-supplied data
is itself useful mathematics, model that check as a normal domain operation
with a specific postcondition and admission rule—not as a companion lifecycle
or generic verification service for computed results.

The domain function may compose a maintained backend such as SymPy, FLINT,
NetworkX, or Z3 where that algorithm is relevant. Those backends remain private
computational engines behind Jacobian's public mathematical contracts.

## Transport and mathematical ownership

The MCP Python SDK owns the transport boundary: registration of `math.find` and
`math.run`, their outer argument and output schemas, protocol validation, and
structured JSON delivery. Jacobian does not duplicate those
checks. The SDK's Streamable HTTP request-body ceiling is an input constraint;
it does not define a tool-result byte ceiling. No MCP response-size limit is
therefore inferred from the canonical codec's defaults.

Mathematical values enforce canonical representation and intrinsic
representation bounds, never JSON response bytes. Operation owners bound work,
intermediate growth, and unavoidable result cardinality. Native functions
return their exact typed values without inheriting an MCP or JSON byte budget.
If a deployment adds a real delivery ceiling, the MCP adapter owns and applies
that ceiling explicitly. Canonical encoding is deterministic measurement by
default and enforces output bytes only when its caller supplies limits for a
concrete boundary.

`math.run` still needs a small dispatch boundary because
its `payload` has an operation-specific schema that is known only after its
immutable `operation_id` is resolved. Dispatch therefore does only this:
resolve the declaration, parse the payload once with that owner's request model,
invoke that owner once, and project the typed result once. The internal
`execute_operation` seam owns this complete envelope for `math.run`, direct
operation tools, and native dispatch; only the final projector differs. The
request context remains bound through canonical projection. It does not contain
domain admission, backend logic, result-specific replay, workflow state, or a
transport byte ceiling. An HTTP request-body limit is enforced before shared
dispatch; stdio and in-process composition do not inherit it.

Owners call `request_checkpoint(stage)` immediately after an external or
backend return and at documented bounded intervals in long native loops. The
same primitive observes both the bound cancellation signal and the current
request deadline, so owners do not maintain local variants of deadline and
cancellation policy.

Rejections retain the phase that owns them:

| Phase | Python boundary | MCP projection |
| --- | --- | --- |
| JSON canonicalization or structural Pydantic parsing | `OperationRequestValidationError` | `INVALID_PARAMS` |
| Native mathematical admission | `OperationDomainValidationError` | `INVALID_PARAMS` |
| Timeout or cancellation | Typed execution exception | Tool error (`is_error=true`) |
| Worker, host, transport, or backend failure | Operational exception | Tool error (`is_error=true`) |

Dispatch does not turn native admission into structural request validation.
MCP deliberately projects both validation classes through
`INVALID_PARAMS` because both mean that the selected operation cannot accept
the supplied payload. Capacity is different from validity: an admitted request
may still fail on a particular worker, host, or delivery boundary. MCP returns
that non-completion as an agent-visible tool error. Timeout, cancellation,
resource exhaustion, and unexpected execution failure establish no
mathematical conclusion; Jacobian does not need a universal capacity exception
hierarchy to state that rule.

Jacobian is a typed, bounded tool layer over maintained mathematical libraries.
The runtime ownership rule above keeps repeated mathematical work out of
ordinary execution and deserialization.

Domain values live beside the functions that own their semantics under
`jacobian.math.<domain>`. HNF, LLL, and Smith-related direct computations call
maintained backends in process; a subprocess is retained only where actual
external isolation is required.

Each mathematical owner keeps its public values and functions in ordinary
semantic modules, private Pydantic wire models in `_models.py` where needed,
and its immutable `TOOLS` tuple in `_tools.py`. That tuple is the owner-local
publication manifest: declared tools are public, while useful native-only
functions remain ordinary package exports without a `MathTool` declaration.
Catalog construction discovers packaged `_tools.py` modules under
`jacobian.math`, sorts their module paths, validates every manifest and
operation ID, and freezes the resulting inventory. There is no parallel
decision ledger, central domain list, or external plugin discovery.

Private wire request and response models may contain operation parameters and
domain-owned mathematical values. They do not justify parallel native and wire
classes for the same mathematical value. A shared value can retain exact
integers in Python and encode those fields as decimal strings only for JSON;
see [native integer codec requirements](../reference/value-interoperability.md#requirements-for-a-native-integer-codec).

`jacobian.catalog` owns declaration models, search, and immutable lookup;
`jacobian.dispatch` owns strict invocation;
`jacobian.mcp` and the CLI are delivery boundaries. The private root model and
exact-scalar helpers contain only behavior genuinely shared by unrelated
owners.

An immutable declaration may carry a small `discovery_terms` vocabulary of
reviewed, established names for its exact postcondition. Terms are catalog
metadata used by deterministic `math.find` ranking; they do not alter the
canonical title and description, operation ID, request syntax, or mathematical
claim. This keeps ordinary morphology in the shared lexical normalizer and
domain terminology with the owner that can review its meaning.

Catalog publication is not runtime planning. The
mathematical owner decides request admission, builds a request-scoped execution
plan when one is useful, owns the backend adapter, and constructs the canonical
result.
Defining-invariant evidence belongs in the operation's tests; a full replay is
not part of ordinary execution. An adapter may reject malformed backend data
while converting it, but that is integration safety rather than a separate
mathematical result stage.
After owner admission succeeds, dispatch and MCP project the typed result for
delivery. A configured delivery limit may still fail operationally, but it
does not retroactively make the mathematical request invalid.

## Bounded worker adapters

Use a child process only for a concrete isolation, killability, or fixed-toolchain
need. The mathematical owner remains responsible for the complete request
envelope: it admits the request, retains its canonical source, starts the
worker, and constructs the final result. The worker receives one strict payload
and returns only a bounded derived projection. The parent binds that projection
to its admitted source before trusted result construction; a worker does not
echo or replace retained canonical values. The parent never passes worker output
through the complete public result model's validation path: it decodes only the
projection's bounded structure, constructs trusted nested values, and calls the
owner's private factory without replaying the worker's mathematics.

The owner charges parsing, launch, backend work, projection, validation, and
cleanup against one local execution plan and deadline. Worker capture limits
cover the actual UTF-8 projection, not an assumed public-result shape. A new
process owner must be named in both the architecture check and the import
contract, with a concrete backend rationale. The detailed codec, cleanup, and
typed-failure rules live in the [mathematical backend contract](../reference/mathematical-backends.md#child-process-adapters).

## Package organization and family folding

A top-level `jacobian.math.<family>` package names a concrete, recognizable
mathematical subject with coherent canonical values and operations. Prefer
subjects such as matrices, graphs, polynomials, probability, and number theory
over vague umbrellas that would contain much of the library. Keep the top level
free of ticket-shaped feature packages, backend names, and workflow groupings.

The first-level taxonomy is deliberately restrained:

```text
jacobian.math
  matrices            polynomials          finite_fields
  groups              lattices             graphs
  combinatorics       number_theory        geometry
  topology            analysis             probability
  optimization        dynamics             logic
```

Independent subjects that do not fit those owners honestly remain explicit:
`cluster_algebras`, `coalgebras`, `crossed_products`, `finite_categories`,
`finite_dim_algebras`, `finite_semigroups`, and `universal_algebra`. Do not
force them into an inaccurate parent merely to reduce the directory count.

Decide by evidence, in this order:

1. **Canonical value ownership.** Each mathematical value has one package that
   owns its meaning, invariants, and public type. Producers and consumers reuse
   that type unchanged.
2. **Operation ownership.** Place an operation with the mathematical subject
   that owns its principal source and postcondition. Returning or accepting
   another family's canonical value does not by itself move the operation into
   that family. For example, a graph characteristic-polynomial operation
   remains graph-owned while returning the polynomial owner's canonical value;
   a chip-firing critical-group operation remains graph-owned while returning
   the group owner's canonical value.
3. **Mathematical subdivision.** Nest a capability when it is recognizably a
   subdivision of its parent, such as `graphs/chip_firing`,
   `graphs/coloring`, or `polynomials/interpolation`. A cross-domain type import
   alone is not evidence for nesting.
4. **Self-contained subject.** Keep a capability top-level when it owns a
   coherent independently recognizable subject rather than specializing an
   existing family.

The first segment of an operation ID (`graph.*`, `matrix.*`, `polynomial.*`,
`formal_series.*`) names the mathematical discovery family even when the
package name does not. It is useful evidence but not sole ownership authority:
never rename operation IDs merely to follow a package move.

Package depth follows mathematical cohesion and implementation complexity, not
a mandatory file template. A small owner may keep its canonical values, native
functions, wire models, and declarations in a few modules. A large owner may
split into mathematical subpackages, each with the private models, bounds,
kernels, backend adapters, declarations, and tests it actually needs. Do not
create empty or pass-through `_bounds.py`, `_operations.py`, `_flint.py`, or
similar files merely to make owners look uniform.

```text
small owner                    large owner
  number_theory/                polynomials/
    __init__.py                    __init__.py
    values.py                      values.py
    operations.py                  ideals/
    _models.py                     interpolation/
    _tools.py                      series/
```

Nest into a subpackage when a cohesive mathematical slice needs its own values,
models, operations, declarations, tests, or substantial private implementation;
use a module for one lone capability. Backend modules such as `_flint.py` or
`_singular.py` live at the narrowest mathematical owner whose conversion,
normalization, context, or failure policy they implement. Drop a now-redundant
family prefix when nesting (`matrix_analysis` -> `matrices/analysis`,
`graph_coloring_ops` -> `graphs/coloring`), and keep descriptive names
otherwise.

The repository uses the taxonomy above today. Future package changes preserve
operation IDs and request/result schemas, keep one mathematical owner per tool,
reuse cross-domain canonical values unchanged, and delete replaced paths in the
same change. Change an owner only when the value owner, operation owner, or
mathematical-subdivision evidence changes; directory-count reduction alone is
not a reason to move a package.

Because Jacobian is pre-stable, a fold updates repository imports and public
examples atomically and does not retain forwarding packages at the former path.
Do not maintain parallel old and new import surfaces, migration registries, or
hard-coded path aliases. Catalog discovery remains recursive and derives owner
modules from packaged `_tools.py` files, so adding a mathematical nesting level
does not require a central package inventory.

Logic follows the same rule. CNF canonicalization and assignment checks are
pure direct operations. SAT and bounded QF SMT-LIB solving use the maintained
Z3 Python binding through bounded owner-local workers.

Remote serving uses the same immutable operation library. Authentication
produces a small request-scoped context. Deployment supplies an immutable
service artifact, configuration, and health checks; rollout, rollback, and
persistence remain deployment-platform responsibilities.
