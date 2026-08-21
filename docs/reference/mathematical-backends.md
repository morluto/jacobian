# Mathematical backend contract

[Documentation home](../index.md)

Jacobian owns each public mathematical contract. A maintained backend may own
the computational kernel, but remains a private implementation detail behind
canonical Jacobian values, complete admission, and typed outcomes.

```text
canonical Jacobian input
  -> backend codec
  -> maintained mathematical kernel
  -> strict result conversion
  -> canonical Jacobian result
```

## Common adapter obligations

Every adapter records the supported backend and version range, converts only
already-admitted canonical values, preserves the coefficient domain and parent
identity, and translates expected backend failures into the operation's typed
outcomes. Backend objects and exceptions must not cross the public request or
result boundary.

Automatic generator inference, ambient contexts, and implicit coercion are not
public semantics. A result converter retains every unit, multiplicity, basis,
axis, generator, quotient map, or witness needed by the declared result and
checks its reconstruction or defining invariant where applicable.

## In-process adapters

An in-process adapter has an explicit conversion in each direction, a supported
version range when behavior is version-sensitive, and exhaustive exception
translation for every accepted request. The concrete request model enforces the
backend's coefficient domain, dimensional or degree limits, structural
preconditions, degeneracies, and work bounds before calling it.

## Child-process adapters

A child-process adapter has the same obligations plus:

- a strict, non-evaluating input and output codec;
- executable discovery and supported-version enforcement;
- wall-time and process-output limits;
- request-scoped temporary resources and process cleanup; and
- distinct typed unavailable, timeout, cancellation, and execution-error
  outcomes.

Child processes use the shared bounded-process supervisor. Backend adapters test
their codec and outcome projection; the supervisor's owning tests prove
process-group termination and descendant cleanup.

## Singular

The commutative-algebra domain uses Singular as a private child-process backend
for exact ideal operations. The adapter supplies an explicit rational
polynomial ring, collision-proof internal identifiers, a fixed ordering, and a
strict result encoding. Singular does not define the public request or result
types.

SageMath is useful during development as an independent differential oracle
for Singular-backed algorithms. It is not a Jacobian runtime dependency or a
required CI environment. Required mathematical evidence belongs in bounded,
repository-owned property tests so that it remains deterministic and locally
reproducible.
