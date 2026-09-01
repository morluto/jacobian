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

### Reusing a worker projection as IPC

A compact worker projection can also serve as an internal IPC protocol when
that is the same boundary the operation already needs. Document its framing,
version, byte and structural bounds, source binding, and typed failure
semantics at the boundary. Keep the projection narrower than the public result
and construct the public value in the parent from the retained canonical
source. Do not make an internal projection a second public schema merely
because another process can parse it.

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

## QEPCAD

The plane real-algebraic owner uses
[QEPCAD B 1.74](https://www.usna.edu/Users/cs/wcbrown/qepcad/B/QEPCAD.html)
as a private child-process backend for exact connected components of bounded-size
sign tables in two variables. The required CI lane installs Debian's pinned
`qepcad=1.74+ds-5` package and checks the backend version before running the
owner and process tests.

The adapter first requests a full sign-invariant CAD of `R^2` for the declared
set alone. It then invokes QEPCAD's two-dimensional closure operation once for
each true cell, with every other truth value reset to undetermined. This
satisfies `closure2d`'s full-CAD precondition and yields the exact true-cell
adjacency relation used to choose the canonical component representatives.

When samples are present, a second bounded CAD adds their coordinate minimal
polynomials and the source representatives as tautologies. These polynomials do
not change the declared set, but force every named point onto a zero-dimensional
cell. The adapter maps the refined component relation back to the source-only
representatives, so adding samples cannot change the component profile or its
IDs. The extra projection family is checked against the same degree,
Sylvester-determinant coefficient-height, and cell envelopes before the second
CAD begins. Plane points reuse the canonical
real-algebraic scalar for each coordinate and retain one rational isolating box.

Degree sixteen is the exact coordinate-carrier bound for this source domain.
A maximal-dimensional two-cell has a rational sector sample. For a
one-dimensional component, the selected cell is either a section of a quartic
over a rational base sector or a sector over a root of a source coefficient,
discriminant, or pairwise resultant, whose degree is at most sixteen. An
isolated zero-dimensional intersection of quartics has local Bezout degree at
most sixteen; an isolated point on a shared reduced curve is singular and has
the sharper degree-twelve derivative bound. The system
`y^4 - x = x^4 - 2 = 0` attains degree sixteen in the `y` coordinate, so the
carrier cannot remain at degree eight.

One bounded Python worker owns sample recognition, QEPCAD interaction, cell
closure, and canonical representative conversion. QEPCAD's prompt exchange is
adaptive but request-scoped: its child joins that worker's process group, and a
callback-scoped byte channel carries each command and response under the same
absolute deadline. No reusable process session survives the request. Each
response frame is limited to 8 MiB, and one 64 MiB stdout ledger is shared by
the source CAD and optional sample-classification CAD rather than renewed for
each cell closure. Degenerate empty and whole-plane requests use the same
killable worker for sample recognition but do not require QEPCAD.

Representative coefficient height, isolating-rational height, one-cell sample
text, refinement-formula size, and aggregate canonical result size are
declared operational completion envelopes rather than restrictions on the
accepted semialgebraic set. A large sector rational or algebraic expression can
therefore exhaust one of those envelopes after request admission. Such a call
returns `RESOURCE_LIMIT` with no component count, representative, or other
topological conclusion; it never truncates an exact profile.

The parent retains the canonical request, strictly decodes only a compact
representative/component-ID projection, binds its axes and sample count back to
that request, and constructs the public result. Worker timeout, output or cell
exhaustion, unsupported version, malformed output, and execution failure remain
distinct operational non-completions and never imply a topological conclusion.
