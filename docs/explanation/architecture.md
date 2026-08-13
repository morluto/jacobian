# Architecture

[Documentation home](../index.md)

- Status: Current architecture

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

Declared input ports may also bind opaque runtime-local value references. The
reference carries no assurance or serialized type metadata: the runtime checks
its stored type and source port, assembles the request, and parses it exactly
once. Closing the runtime invalidates its references.

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
Linter enforces the leaf boundaries and dependency direction. Operation
declarations live outside `jacobian.math`.

The portfolio composition root is the only owner that assembles runtime
services and installation order. Ordinary `DomainBundle` values contain typed
operation declarations only; they do not carry installer callbacks or runtime
collaborators. A capability family with a genuinely specialized artifact or
checker lifecycle is represented as an explicitly named managed portfolio
component rather than widening the semantic bundle contract.
An operation may still bind a typed computational backend, provided that
backend owns no runtime, storage, publication, installation, or checker
authority. This is execution dependency injection, not lifecycle ownership.

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
entry-point discovery are not supported.

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

Pydantic owns complete request validation, including relationships among
otherwise valid fields. Required agreement of parents, characteristics,
presentations, axes, bases, labels, and bound identities is checked during the
single request parse, before preflight or any provider call. JSON Schema is
generated for discovery and is not executed as an additional validation pass
for built-ins. Provider and subprocess output is a separate untrusted boundary
and is parsed independently.

Preflight distinguishes supported, unsupported, provider unavailable, and
resource-limit-exceeded outcomes. Where practical it estimates work, output
size, publication, checker replay, and aggregate allocation before work begins.

A postcondition runs before publication. Failure exposes no value reference,
artifact, or verification record. Terminal execution state remains separate from the
mathematical result and from verification authority.

`CapabilityResult` is a wire projection, not an in-process return type. Domain
functions, operation executors, artifact services, and checker services return
their owned typed values or terminal states. The public dispatcher constructs
the wire envelope once, after publication has returned the complete artifact
closure; installed adapters return a typed projection rather than constructing
the envelope themselves. Artifact-producing operations must not move storage writes
into `OperationSpec.execute`; a domain-specific publisher may preserve an
established durable schema and parent closure without expanding the generic
publication policy.

The v2 wire envelope carries a top-level `verification_record_uri` when an
independent checker accepted the result. That pointer is not an input to
execution, discovery, publication, or mathematical identity. Ordinary
operations do not manufacture generic scope, completeness, relationship, or
obligation records around an already-typed result.

## Verification

A checker is a separate installed operation governed by a typed
`VerificationProtocol[SubjectT, CandidateT, EvidenceT, DecisionT]`. Operator
configuration authorizes checker identities; producer declarations and search
code cannot authorize themselves. Exact replay declarations bind their own
provider-runtime factory. Installation groups those factories by the provider
identity they probe; it does not keep a central entrypoint map or support
matrix.

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

Checker identity comes from a versioned manifest for that checker, not from a
digest of the whole Jacobian package. The manifest binds its exact entry point,
separate checker and worker source closures, exact Python distributions, Python
and provider runtime, passive contracts, and bounded-process policy. The worker
admits only the declared first-party closure and manifest-bound third-party
distributions, including imports requested dynamically during checker
execution, and remeasures the complete
manifest—including the current bytes of every indexed dependency file—around
execution. Authorization performs the same measurement once; catalog and
compatibility reads do not repeat that filesystem scan. A producer or unrelated
checker edit therefore cannot change the identity, while a changed executable
dependency cannot retain it.

Verification record v4 snapshots that complete manifest and binds its canonical
digest, so interpreting the checker identity never depends on a mutable
authorization row. Record v3 belongs to state revision 10 and remains readable
with the matching older checkout; the current runtime has no dual record shape.

`VerificationResult` is the internal typed outcome of that checker execution,
not a generic mathematical result envelope. Capability adapters project it
once into the ordinary operation response. Ordinary producers do not use it,
so checker input validity, conclusions, and evidence bindings do not become
knobs on every mathematical value.

## Values, carriers, and composition

Composition distinguishes four identities:

```text
ValueType       constructor, version, canonical semantics
BoundValueType  exact parent/presentation, axes, labels, bases
ValueInstance   exact semantic value and canonical digest
ValueReference  opaque carrier, source port, provenance
```

The supported internal port contract is deliberately small: an input port
names one typed request field, and an output port exposes the whole typed
result. Pydantic retains field cardinality and cross-field constraints.
Installation verifies each declared field and result type. Compatibility
requires exact type/version, parent, presentation, axes, and bases; all
transformations are explicit.

Finite-field direction ledgers, finite map tables, fiber partitions, and
certificates use this contract unchanged. They remain domain-owned semantic
values, so the port layer needs no collection model, field extraction,
cardinality language, coercion graph, or generalized unifier. A complete
projective line owns its presentation, axis, order, completeness check, and
digest; it does not make ports collection-aware.

`value://opaque-id` tokens contain no serialized metadata. The bounded in-memory
store is owned by one local or tenant runtime and records the semantic value,
source operation and version, output port, and canonical digest. Resolution
checks the exact value class and then lets the assembled Pydantic request check
its presentation, parent, axes, and bases. The store evicts least-recently-used
references to stay within its fixed count and byte bounds; closing the runtime
invalidates everything that remains. Durable promotion is not a supported
operation; durable artifacts remain an explicit, separate publication path.

Using a request-local carrier does not change a value or grant assurance.

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
are lazy even though the maintained Python backends are exact base-package
dependencies. A missing or mismatched maintained Python backend is a broken
installation and fails runtime construction; optional native/formal providers
remain absent without affecting unrelated installed operations. Provider
identity is invocation provenance rather than mathematical value identity.

## Host and protocol boundaries

The local server contains catalog, execution, storage, provider declarations,
and checker authority. Remote authentication, tenants, admission, leases,
eviction, and quarantine belong to a separate remote host and do not enter the
local mathematical server.

Local artifact storage retains one concrete filesystem CAS with SQLite metadata.
`ArtifactRepository` is its public aggregate; explicit transaction, blob, and
metadata collaborators own the implementation. They are not interchangeable
backend interfaces, and new storage abstractions require new workload evidence.
The aggregate coordinates lifecycle and transactions; it does not mirror
collaborator-private blob or recovery APIs.

MCP uses SDK-derived typed schemas and structured output. Pydantic result models
are returned directly unless a genuine `ResourceLink`, custom metadata, or
deliberate text projection requires an explicit MCP result. The two fixed tools
are statically registered; dynamic operation tools and compatibility aliases
are not supported.

Hosting has two constructors with separate ownership: `server.create_server`
owns one concrete local runtime, while `remote.create_remote_server` owns
authentication and isolated tenant runtimes. Their only shared request boundary
acquires one runtime lease; local state has no nullable tenant router, and remote
admission, eviction, and quarantine do not enter the local server module.
The `jacobian-mcp` entry point is local stdio only. Remote transports use the
separate `jacobian-remote-mcp` operator entry point.

The CLI projects the same installed declarations and execution path through
`catalog`, `inspect`, and `run`. Operator administration remains separate from
the mathematical catalog.
