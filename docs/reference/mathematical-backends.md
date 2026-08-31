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

An unexpected backend or host failure never establishes a mathematical
predicate. An adapter may return ``False``, ``UNBOUNDED``, ``UNSAT``, or another
mathematical outcome only from the defining computation; unexpected failures
propagate to the owner for typed operational-failure translation.

Automatic generator inference, ambient contexts, and implicit coercion are not
public semantics. A result converter retains every unit, multiplicity, basis,
axis, generator, quotient map, or witness needed by the declared result and
may reject malformed backend representation. Defining-invariant evidence
belongs in the owning tests or in an explicit bounded verifier when the public
contract accepts independently supplied result data; it is not a universal
converter obligation.

## In-process adapters

An in-process adapter has an explicit conversion in each direction, a supported
version range when behavior is version-sensitive, and exhaustive exception
translation for every accepted request. The shared domain admission path
enforces the backend's coefficient domain, dimensional or degree limits,
structural preconditions, degeneracies, and work bounds before calling it. A
wire request model may invoke that path after parsing; native callers use the
same domain function directly.

### Owner-local adapter placement

Keep an in-process backend private to the mathematical owner whose operation
uses it:

```text
owner-local request admission
  -> owner-local private backend adapter
  -> maintained backend kernel
  -> canonical owner-defined result
```

The owner decides the backend's admitted domain, converts canonical values in
both directions, normalizes backend output, and translates expected failures.
Do not introduce a repository-wide backend facade that mirrors a maintained
library's API or pass backend objects between mathematical owners. Shared
helpers may own genuinely canonical scalar or value construction, but not an
operation's mathematical policy.

Load a native backend lazily when importing the public Jacobian namespace does
not require it. For a single call with no substantial conversion or failure
policy, keep the import and call together in the owner's private operation
path:

```python
def _integer_rank(rows: tuple[tuple[int, ...], ...]) -> int:
    from flint import fmpz_mat

    return int(fmpz_mat(rows).rank())
```

A separate private adapter module is warranted when it owns meaningful
conversion, normalization, backend context, exception translation, or reuse.
For example, an owner may place this boundary in ``_flint.py``:

```python
from fractions import Fraction


def determinant_and_rank(
    rows: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, int]:
    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in rows]
    )
    determinant = backend.det()
    return (
        Fraction(int(determinant.numerator), int(determinant.denominator)),
        int(backend.rank()),
    )
```

The operation computes admission before calling this adapter and constructs its
canonical result after the adapter returns. The adapter is a boundary around
backend-specific mechanics, not an interchangeable-backend framework. Mutable
backend context, including global precision, requires explicit request-lifetime
and concurrency ownership rather than an unguarded set-and-restore sequence.

## Child-process adapters

A child-process adapter has the same obligations plus:

- a strict, non-evaluating input and output codec;
- executable discovery and supported-version enforcement;
- wall-time and process-output limits;
- request-scoped temporary resources and process cleanup; and
- distinct typed unavailable, timeout, cancellation, and execution-error
  outcomes.

The parent owns admission, the retained canonical source, and final result
construction. A worker returns a compact derived projection only; it must not
echo or replace source values. Bind its projection to the admitted parent source
before trusted result construction. Structurally decode the projection, but do
not pass worker output through the complete public result model or any nested
validator that replays mathematical work. Size stdin and stdout limits for the
actual UTF-8 worker payload, not for a different public representation.

Child processes use the shared bounded-process supervisor. Backend adapters test
their codec, source binding, and outcome projection; the supervisor's owning
tests prove process-group termination and descendant cleanup. Register each
process owner in the architecture check and Import Linter exception list; these
are narrow ownership declarations, not a general math-to-process dependency.

The supervisor's wall deadline starts at adapter entry and is shared by input
spooling, launch, resource setup, capture, execution, conversion, and result
delivery. Cleanup may use a separately named finite reaping grace, but it must
be included in the documented maximum return envelope rather than resetting
the operation deadline.

## External reference oracles

A mathematical implementation used only to compare results during development
is a differential oracle, not necessarily a Jacobian runtime backend. Heavy or
native software may remain outside the ordinary installation and required CI
environment when its installation or runtime cost is disproportionate to the
bounded operation under test.

Required correctness evidence must remain deterministic and runnable without
the optional oracle. For a non-unique result, define a canonical normalization
or mathematical equivalence before comparing implementations; incidental
ordering, rooting, identifiers, or witness choice are not disagreements.
Agreement with an oracle supplements but never replaces Jacobian's independent
reconstruction or defining-invariant validation.

## Singular

The commutative-algebra and polynomial-map domains use Singular as a private
child-process backend for exact ideal and generic-fiber operations. The
adapters supply an explicit rational polynomial ring, collision-proof internal
identifiers, a fixed ordering, and a strict result encoding. Generic-fiber
degree evidence also carries a lift matrix back to the source ideal and has an
explicit independent verification pass. Singular does not define the public
request or result types.

SageMath is the current development-time differential oracle for selected
Singular-backed algorithms. It is not a Jacobian runtime dependency or a
required CI environment; the Singular adapter and bounded repository-owned
property tests carry the required evidence.
