# Product model: atomic mathematics for agents

[Documentation home](../index.md)

- Status: Active product direction
- Scope: MCP, CLI, and Python access to installed mathematical operations

## Product definition

Jacobian is a toolbox of atomic mathematical operations for agents. The agent
owns decomposition, sequencing, checker choice, and stopping. Jacobian owns
typed mathematical boundaries, resource admission, operation installation, and
operator authorization of independent checkers.

```text
math.find   search or inspect an installed operation
math.run    run one installed operation and return a value or checker verdict
```

The full installed inventory is available at `capability://catalog`. Adding a
mathematical operation does not add an MCP tool.

An ordinary operation has one mathematical outcome: compute a determinant,
factor a polynomial, enumerate a bounded family, or construct a separating
hyperplane. A checker is another operation with its own catalog ID. Successful
computation never authorizes its own result as independently verified.

## Ownership model

| Owner | Responsibility |
| --- | --- |
| Agent | Representation choices, multi-step strategy, checker selection, stopping |
| `jacobian.math` | Public mathematical values, constructors, and functions |
| Operation declarations | Typed request/result binding and mathematical pre/postconditions |
| Runtime | Installation, provider selection, execution bounds, publication, and provenance |
| Operator | Checker authorization and host policy |
| MCP SDK | Static tool schemas, typed structured output, transport, progress, and cancellation |

The dependency direction is one way:

```text
MCP / CLI / hosts
        │
        ▼
runtime and installed operation bindings
        │
        ├──► PublicationPolicy + ProviderBinding
        │
        └──► OperationSpec ──► jacobian.math.<domain>
        │
        ▼
private maintained backends
```

`OperationSpec` owns semantic operation metadata. An installed binding pairs it
with provider selection and a separate publication policy. Publication owns
transport only; it does not own mathematical validation, applicability,
provider selection, effects, checker authority, or request parsing.

## Mathematical values

Provider-independent mathematical identity belongs to the owning domain. The
small `jacobian.contracts` package is reserved for genuinely cross-domain
passive primitives such as digests, nominal references, exact scalars, bounded
collections, and transport-neutral reference primitives.

Domain values live beside domain functions, for example:

```text
jacobian.math.matrices.values
jacobian.math.polynomials.values
jacobian.math.graphs.values
jacobian.math.finite_fields.values
jacobian.math.linear_maps.values
```

Value modules do not import providers, runtime, storage, MCP, installation, or
checker authority. Public domain packages re-export only their supported
values, constructors, and functions through explicit `__all__` values.

Every public mathematical function accepts one canonical semantic input type.
That type may be a Python scalar, a maintained backend type whose object already
carries the complete semantics, or a Jacobian-owned value when parent,
presentation, axes, labels, basis, ordering, normalization, canonicalization,
role, or evidence binding would otherwise be missing.

```python
gcd(12, 18)
resultant(sympy.Poly(..., domain=QQ), sympy.Poly(..., domain=QQ))
A = matrix([[1, 2], [3, 4]], domain=ZZ)
rank(A)
```

Interoperability is explicit (`Matrix.from_sympy`,
`Polynomial.from_sympy`, `Graph.from_networkx`). Backend objects are never wire,
artifact, or cross-provider composition identity merely because a backend can
compute with them.

## Values, carriers, and records

A mathematical value is distinct from how it travels. An inline value, an
opaque request-local `value://` reference, and a durable `artifact://` carrier
must resolve to the same semantic value and digest when all three are allowed.
Changing carrier grants no assurance.

Invocation records describe execution and provider provenance. Verification
records bind an accepted checker decision to the exact subject, candidate,
evidence, semantics, scope, certificate format, and checker identity. Neither
record is the mathematical value.

Small bounded values stay inline. Durable artifacts are reserved for identity,
independent retrieval, replay, resumability, evidence binding, or
size-separated transport. Ordinary computations do not expose a generic
persistence flag.

## Search, execution, and checking

`math.find` has two purposes: lexical search and exact inspection. Search may
report factual applicability, provider availability, checker availability, and
cost admission. It never recommends a workflow or a next operation. The full
inventory remains a resource rather than an empty-query search mode.

`math.run` executes one selected ID. The external request is parsed once,
preflight runs before allocation, one semantic function executes, a typed
request-to-result postcondition runs before exposure, and the result is
serialized once. Timeout, cancellation, provider failure, resource refusal,
and checker interruption are non-conclusions.

Checker operations remain independent and operator-authorized. Availability is
not authorization, exact arithmetic is not independent verification, and a
failed search is not a negative theorem.

## Product boundaries

Jacobian is not:

- a research-workflow engine;
- a claim decomposition or conjecture-management service;
- a plugin execution framework;
- a generic witness, transformation, shrinking, or experiment service;
- a universal solver or backend wrapper;
- a second semantic type system above maintained libraries;
- one MCP tool per mathematical operation; or
- a mandatory explore/verify sequence.

Worked investigations belong in scenarios and benchmarks. Harbor tasks, hidden
verifiers, and operator-run model evaluations are evaluation infrastructure,
not runtime workflow features.

External operation packages remain unsupported until a concrete package proves
the need. Discovery, enablement, provider availability, and checker authority
will remain separate decisions if that surface is introduced.

## Architecture budgets

A shared abstraction must replace repetition in at least two surviving
production paths in the same change. An ordinary operation should need no more
than one public domain function, one request model when necessary, one rich
result type when necessary, one `OperationSpec`, and one external publication
binding only when inline transport is insufficient.

Transforms such as transpose, embedding, basis change, restriction of scalars,
reduction, permutation, projection, and reindexing are explicit mathematical
operations. Compatibility, references, persistence, and provider identity
never grant verification assurance.
