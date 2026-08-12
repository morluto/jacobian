# Domain operation library

[Documentation home](../index.md)

- Status: Current library contract
- Related architecture: [Domain operation library](../explanation/architecture.md#domain-operation-library)

`jacobian.math` is Jacobian's authoritative mathematical library. Built-in
catalog operations are typed bindings over that library, not a second
implementation and not a facade invoked by the native API.

This reference defines ownership and dependency rules. The live
`capability://catalog` remains the authority for installed IDs, schemas, and
provider availability.

## Package ownership

Each domain owns values, constructors, public functions, and private backend
conversion:

```text
jacobian/math/matrices/
    __init__.py
    values.py
    operations.py
    _sympy.py

jacobian/math/finite_fields/
    __init__.py
    values.py
    operations.py
    _sympy.py
    _flint.py
```

Public packages declare explicit `__all__` values. Value modules import no
provider, runtime, storage, MCP, operation installation, publication, or
checker-authority code. Private backend modules may import maintained
third-party mathematics and remain lazy when the provider is optional.
They stay small: backend calls, exact conversions, and backend-specific
normalization belong there, but generic adapters, mirrored backend APIs, and a
second wrapper type system do not. Public functions validate the documented
semantic contract and delegate to these maintained libraries.

**Wrap semantic contracts, not entire libraries.** Add a private backend
boundary only when Jacobian must preserve semantics, convert exact values,
normalize results, isolate an optional import, or contain a backend-specific
quirk. A stable backend call that needs none of those may remain a direct call;
do not reproduce a maintained library's API under Jacobian names.

The cross-domain `jacobian.contracts` package is limited to passive primitives:
digests, nominal reference identifiers, exact scalars, common bounded
collections, and transport-neutral reference primitives. Values whose parent,
presentation, axes, basis, ordering, role, or evidence binding changes meaning
belong to their mathematical domain.

## Canonical semantic inputs

Every public function has one canonical semantic input type. Use a Python
scalar when it is complete, a maintained backend type when the backend object
already fixes every semantic fact, or a Jacobian-owned immutable value when it
does not.

Examples:

```python
gcd(12, 18)
resultant(sympy.Poly(..., domain=QQ), sympy.Poly(..., domain=QQ))
A = matrix([[1, 2], [3, 4]], domain=ZZ)
rank(A)
```

Do not expose broad overload sets across lists, dictionaries, mutable backend
objects, and Jacobian values. Explicit constructors such as
`Matrix.from_sympy`, `Polynomial.from_sympy`, `Graph.from_networkx`, and
`FiniteFieldElement.from_flint` own interoperability.

Canonical decimal strings are wire and persistence values rather than
computational values. Boundary code uses the canonical conversion API before
constructing backend inputs or result values. Internal JSON round-trips and
direct `int()` or `str()` conversion of canonical scalar components are not
supported.

## Semantic operation declarations

Operation declarations live outside `jacobian.math`. The initial shared form
is deliberately small:

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

`execute` may bind request fields to a public function, for example
`lambda request: determinant(request.matrix)`. It may not reimplement the
mathematics.

An installed operation pairs the semantic declaration with separate provider
and publication facts:

```python
@dataclass(frozen=True, slots=True)
class InstalledOperation(Generic[RequestT, ResultT]):
    spec: OperationSpec[RequestT, ResultT]
    publication: PublicationPolicy[ResultT]
    provider_binding: ProviderBinding
```

Publication owns inline bounds, previews, request-local references, durable
artifacts, and semantic closure over referenced parents and axes. It never owns
request validation, mathematical postconditions, domain applicability,
provider selection, checker authority, or operation effects.

`DomainBundle` groups ordinary installed operations and their shared semantics;
it has no managed-installer callback. Specialized artifact/checker lifecycles
belong to explicitly named components in the portfolio composition root. This
keeps runtime services and dependency ordering out of the semantic operation
model instead of adding an optional installation mode to every bundle.

## Execution contract

The ordinary path is:

```text
external payload
  → resolve InstalledOperation
  → one TypeAdapter parse
  → preflight
  → jacobian.math function
  → request/result postcondition
  → Completed | NonConclusion | Failed
  → publication
  → one serialization
```

Pydantic validates the complete request before computation or artifact writes.
JSON Schema supports discovery and is not an extra built-in execution pass.
Provider or subprocess output is separately parsed because it crosses another
untrusted boundary.

Preflight distinguishes `SUPPORTED`, `UNSUPPORTED(reason)`,
`PROVIDER_UNAVAILABLE`, and `RESOURCE_LIMIT_EXCEEDED`. It includes bounded work,
output, publication, replay, and aggregate allocation estimates where those are
known without executing the operation.

A request-to-result postcondition runs before anything is exposed. Failure
publishes no value reference, artifact, or verification record.

Envelope metadata is not part of `OperationSpec`. In particular, an
ordinary operation does not configure generic completeness, scope,
relationships, or obligations. A v2 result carries a `verification_record_uri`
only when an independent checker accepted the result; that pointer carries no
mathematical or verification authority beyond the record it identifies.

## Bounded searches

A retained bounded search is an ordinary `OperationSpec`. Its domain-owned
result records the facts callers need—for example `EXACT`, `INCOMPLETE`, or
`UNKNOWN`, the admitted bounds, and any witness. It does not acquire generic
scope, completeness, relationship, or obligation wrappers.

`Completed` means the bounded computation returned a valid typed result; it
does not imply exactness unless that result says so. Timeout, cancellation,
provider error, resource refusal, and failure to produce a contract-valid
result are artifact-free non-conclusions. A separate checker request contains
the exact subject and candidate it replays. There is no shared `SearchSpec` or
`BoundedSearchResult` until at least two surviving operations prove that a
domain result cannot express their common semantics.

## Independent checkers

A checker is a separate installed catalog ID with a typed
`VerificationProtocol[SubjectT, CandidateT, EvidenceT, DecisionT]`.
Operator configuration owns authorization. Producer declarations, provider
availability, search, and plugins cannot authorize a checker.

Checker execution may share passive schemas, constants, format specifications,
and conformance vectors with a producer. It does not share executable proposal,
search, conversion, or mathematical replay code with the path it certifies.

Accepted replay may commit a verification record bound to the exact subject,
candidate, evidence, protocol, semantics, scope, certificate format, checker
identity, and runtime identity. Rejection is a checker verdict; interruption,
timeout, provider failure, malformed output, or missing evidence is a
non-conclusion and cannot create a record.

An inline producer may expose its whole typed result through an output port and
an exact checker may accept that value through a candidate input port. This
changes only the runtime-local carrier: the checker still parses the assembled
typed request once, independently replays the relation, and alone owns any
verification record. A candidate reference never transfers producer authority.

## Values and publication

Small bounded values remain inline. Use a request-local reference or durable
artifact only when the outcome needs identity, independent retrieval, replay,
resumability, evidence binding, or size-separated transport. There is no
generic persistence flag.

Inline values, `value://` references, and `artifact://` carriers resolve to the
same semantic value and digest when each carrier is permitted. Publication,
provider provenance, invocation records, and verification records remain
separate from mathematical value identity.

## Adding an operation

An ordinary operation should require at most:

1. one public domain function with one canonical input type;
2. one request model when a boundary model is necessary;
3. one rich result/value model when a scalar is insufficient;
4. one `OperationSpec`; and
5. one external publication binding only when inline transport is insufficient.

Add focused behavior tests through the public Python function and installed
operation seam. Verify canonical/backend round trips where conversions exist,
producer-to-consumer compatibility where values compose, and import isolation.
When a checker is justified, test its independent replay and fail-closed record
binding separately.

Do not add a shared abstraction until two surviving production paths use it and
the older duplication is deleted in the same change. Do not add global
registries, recursive discovery, mechanical backend wrappers, automatic
coercion, a generic conversion framework, or paper-shaped combined operations.
