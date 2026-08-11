# Domain operation library

[Documentation home](../index.md)

- Status: Target contract for the pre-stable cutover
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
```

Public packages declare explicit `__all__` values. Value modules import no
provider, runtime, storage, MCP, operation installation, publication, or
checker-authority code. Private backend modules may import maintained
third-party mathematics and remain lazy when the provider is optional.

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
publishes no value reference, artifact, or assurance.

Legacy envelope metadata is not part of `OperationSpec`. In particular, an
ordinary operation does not configure generic assurance, completeness, scope,
relationships, or obligations. Any temporary v2 response label is an outward
compatibility projection and carries no mathematical or verification
authority.

## Bounded searches

A retained bounded search is an ordinary `OperationSpec`. Its domain-owned
result records the facts callers need—for example `EXACT`, `INCOMPLETE`, or
`UNKNOWN`, the admitted bounds, and any witness. It does not acquire generic
scope, completeness, relationship, obligation, or assurance wrappers.

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
