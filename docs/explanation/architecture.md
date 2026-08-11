# Architecture

[Documentation home](../index.md)

- Status: Target architecture for the pre-stable cutover

## Purpose

Jacobian exposes a growing portfolio of atomic mathematical operations through
exactly two MCP tools:

```text
agent
  ├── math.find ──► search or inspect installed operations
  └── math.run  ──► one operation → mathematical value or checker verdict
```

The agent owns strategy. The server owns typed execution, resource admission,
publication, provider provenance, and checker authorization.

## Search and execute

`math.find` performs lexical search or exact inspection. Search results are
factual projections of installed declarations. Typed compatibility and
preflight may narrow those results, but discovery does not plan a sequence or
recommend a next operation.

`capability://catalog` is the complete inventory. Search is not an alternate
inventory protocol, and an empty query is not browse mode.

`math.run` accepts an operation ID and payload:

```json
{
  "capability_id": "matrix.determinant.compute",
  "payload": {
    "matrix": {
      "matrix_schema_version": "1",
      "domain": "QQ",
      "entries": [
        [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
        [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}]
      ]
    }
  }
}
```

Once typed composition lands, declared input ports may also bind opaque value
references. The assembled request is still parsed exactly once.

## Dependency direction

```text
MCP / CLI / hosts
    import runtime and public projections

runtime
    imports installed bindings, providers, storage, checker authority

installed binding
    = OperationSpec + PublicationPolicy + ProviderBinding

OperationSpec
    imports jacobian.math values and functions

jacobian.math.<domain>
    imports domain values and may call private backend modules

private backend modules
    import third-party libraries only
```

Private mathematical backends never import runtime, operation declarations,
MCP, CLI, storage, installation, publication, or checker authority. Import
Linter contracts enforce this boundary incrementally as domains migrate and
become exhaustive after the domain cutover. Operation declarations live
outside `jacobian.math`.

## Mathematical value and backend layers

Domain packages own values and functions vertically:

```text
jacobian.math.matrices/
    __init__.py       supported exports
    values.py         provider-independent identity
    operations.py     public mathematical functions
    _sympy.py         private conversion/backend calls
```

Every public function accepts one canonical semantic input type. A maintained
backend type is appropriate when it already fixes every fact that changes the
meaning. Jacobian owns a value when parent, presentation, axes, labels, basis,
ordering, normalization, canonicalization, role, or evidence binding would
otherwise be implicit.

Canonical decimal strings are external and persistence values, not internal
computational values. Boundary code uses the canonical conversion API before
constructing backend values. Mathematical code never round-trips through JSON.

## Domain operation library

`OperationSpec[RequestT, ResultT]` is the small semantic declaration:

```python
@dataclass(frozen=True, slots=True)
class OperationSpec(Generic[RequestT, ResultT]):
    operation_id: str
    version: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    execute: Callable[[RequestT], ResultT]
    preflight: Callable[[RequestT], PreflightResult] | None = None
    postcondition: Callable[[RequestT, ResultT], None] | None = None
    effect: Effect = Effect.READ_ONLY
```

The callable binds validated request fields to one public mathematical
function; it is not another implementation of the mathematics.

An installed binding adds transport and provider facts:

```python
@dataclass(frozen=True, slots=True)
class InstalledOperation(Generic[RequestT, ResultT]):
    spec: OperationSpec[RequestT, ResultT]
    publication: PublicationPolicy[ResultT]
    provider_binding: ProviderBinding
```

Publication may decide inline eligibility, bounded previews, request-local
references, durable artifacts, and closure over referenced parents or axes. It
does not own mathematical validation, applicability, checker authority,
provider selection, effects, or parsing.

Built-ins use a static explicit inventory. External operation packages and
entry-point discovery are not supported during the cutover.

## Execution pipeline

```text
external payload
  → resolve InstalledOperation
  → one TypeAdapter parse
  → preflight
  → execute one semantic function
  → request/result postcondition
  → Completed | NonConclusion | Failed
  → publication projection
  → one serialization
```

Pydantic owns complete request validation. JSON Schema is generated for
discovery and is not executed as an additional validation pass for built-ins.
Provider and subprocess output is a separate untrusted boundary and is parsed
independently.

Preflight distinguishes supported, unsupported, provider unavailable, and
resource-limit-exceeded outcomes. Where practical it estimates work, output
size, publication, checker replay, and aggregate allocation before work begins.

A postcondition runs before publication. Failure exposes no value reference,
artifact, or assurance. Terminal execution state remains separate from the
mathematical result and from verification authority.

The transitional v2 wire envelope may still project a `COMPUTED` label for a
completed ordinary operation or a non-conclusive label for failure. Those are
compatibility fields, not inputs to execution, discovery, publication, or
mathematical identity. Ordinary operations do not manufacture generic scope,
completeness, relationship, or obligation records around an already-typed
result.

## Verification

A checker is a separate installed operation governed by a typed
`VerificationProtocol[SubjectT, CandidateT, EvidenceT, DecisionT]`. Operator
configuration authorizes checker identities; producer declarations and search
code cannot authorize themselves.

The runtime keeps four narrow responsibilities:

- build a plan that binds subject, candidate, evidence, protocol, checker,
  scope, and limits;
- execute the authorized checker within those bounds;
- parse accepted, rejected, or non-conclusion decisions; and
- commit a verification record only for a valid accepted decision.

Timeout, cancellation, malformed output, unavailable providers, interruption,
and failure to find evidence are non-conclusions. The record binds the exact
subject, candidate, evidence, protocol, semantics, scope, certificate format,
and checker identity. Independent checker execution does not import or call the
producer, proposal, search, or evaluation path it certifies.

## Values, carriers, and composition

Composition distinguishes four identities:

```text
ValueType       constructor, version, canonical semantics
BoundValueType  exact parent/presentation, axes, labels, bases
ValueInstance   exact semantic value and canonical digest
ValueReference  opaque carrier, source port, provenance
```

Initial internal ports are deliberately small and provisional through #905
Slice A: an input port binds one typed value to a request, and an output port
extracts one typed value from a result. One port may carry one semantic value or
one bounded homogeneous collection. Pydantic retains field cardinality and
cross-field constraints. Installation verifies each typed accessor against its
declared value type. Compatibility requires exact type/version, parent,
presentation, axes, and bases; all transformations are explicit. The port
contract is frozen only after Slice B demonstrates reuse.

`value://opaque-id` tokens contain no serialized metadata. The request-local
store owns tenant/session lifetime, semantic value, source operation, operation
version, output port, bound identities, canonical digest, and invocation
provenance. Resolution
validates those stored facts against the target port. Durable promotion closes
over every referenced value, parent, basis, and axis.

Changing between inline, request-local, and durable carriers does not change a
value or grant assurance.

## Structural JSON

The runtime retains direct bounded JSON functions rather than a codec
framework: a strict loader, deterministic serializer, Pydantic adapters, and
one explicit schema-only adapter. Generic JSON rejects duplicate keys and
unsupported numbers, enforces depth/member/byte limits, preserves keys and
strings exactly, and performs no Unicode or mathematical normalization.

Type-specific owners handle mathematical encodings such as exact rationals.
Digest-affecting changes require frozen vectors and an explicit state/codec
revision; incompatible old state is rejected instead of served through dual
canonicalizers.

## Bounded searches

A retained bounded search uses the same `OperationSpec` path as any other
mathematical operation. Its domain-owned result carries the applicable status,
bounds, coverage facts, and witness directly. `Completed` means only that this
typed result was produced; exactness is established by the result's own status,
not by a generic completeness envelope.

Timeout, cancellation, provider error, and resource refusal publish no partial
result or obligation artifact. Independent checker operations receive the
typed subject and candidate they need. No `SearchSpec`, generic bounded-result
wrapper, or bounded-operation adapter exists unless two surviving operations
later prove a common semantic need.

## Provider optionality

Importing `jacobian.math` performs no provider probe. Private backend imports
are lazy, optional providers are installation extras, and an unavailable
provider removes only its affected installed operations. Provider identity is
invocation provenance rather than mathematical value identity.

## Host and protocol boundaries

The local server contains catalog, execution, storage, provider declarations,
and checker authority. Remote authentication, tenants, admission, leases,
eviction, and quarantine belong to a separate remote host and do not enter the
local mathematical server.

MCP uses SDK-derived typed schemas and structured output. Pydantic result models
are returned directly unless a genuine `ResourceLink`, custom metadata, or
deliberate text projection requires an explicit MCP result. The two fixed tools
are statically registered; dynamic operation tools and compatibility aliases
are not supported.

The CLI projects the same installed declarations and execution path through
`catalog`, `inspect`, and `run`. Operator administration remains separate from
the mathematical catalog.
