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

When an operation promotes a backend candidate to a mathematical certificate,
the producing kernel must establish that certificate's conditions once before
trusted result construction. A backend status or a second call to the same
solver is not independent certification. For example, infeasibility of
`Ax=b, x>=0` can be certified by `A^T y>=0` and `b^T y<0`; converting the
candidate to exact rationals does not establish those inequalities. An invalid
primal candidate establishes neither infeasibility nor unboundedness. Preserve
an operational-failure outcome unless a valid mathematical result is obtained.

This is part of producing the certificate, not a universal result-verification
layer. Do not replay a solve, factorization, or other completed kernel in a
Pydantic validator, serializer, or result constructor. Consult the
[known backend defects](backend-known-defects.md) when investigating inconsistent
candidates; use a maintained alternative or a bounded owner-local repair, with
regressions for the pinned backend behavior.

Automatic generator inference, ambient contexts, and implicit coercion are not
public semantics. A result converter retains every unit, multiplicity, basis,
axis, generator, quotient map, or witness needed by the declared result and
may reject malformed backend representation. Defining-invariant evidence
belongs in the owning tests. If checking caller-supplied mathematical data is a
public capability, its domain operation owns that check; it is not a universal
converter obligation or backend replay stage.

## In-process adapters

An in-process adapter has an explicit conversion in each direction, a supported
version range when behavior is version-sensitive, and exhaustive exception
translation for every accepted request. The shared domain admission path
enforces the backend's coefficient domain, dimensional or degree limits,
structural preconditions, degeneracies, and work bounds before calling it. A
wire request model handles structural parsing; execution then calls the same
domain function as the native API. Non-trivial admission is not repeated in
request validators.

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

The parent owns admission and final result construction. Every worker boundary
checks object shape, collection cardinality, row widths, and scalar syntax
before materializing nested values. Use the canonical codec when encoded bytes
participate in a digest or size proof; otherwise a strict, deterministic,
non-evaluating codec is sufficient. Mathematical integers that may exceed the
interoperable JSON range use canonical decimal strings, while intrinsically
bounded counters may remain JSON integers.

When a worker returns a derived projection of canonical source retained by the
parent, it must not echo or replace that source. Bind the projection to the
admitted source before trusted result construction. Workers that return a
self-contained bounded value do not need an artificial source digest.
Structurally decode projections, but do not pass worker output through the
complete public result model or any nested validator that replays mathematical
work. Size stdin and stdout limits for the actual UTF-8 worker payload, not for
a different public representation. Pass those channel-specific limits to both
the process supervisor and any canonical encoder or decoder at that boundary;
the codec's ordinary unbounded-output mode must not silently substitute a
different default.

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

The commutative-algebra, polynomial-map, and projective plane-curve domains use
Singular as a private child-process backend for exact ideal, generic-fiber, and
singularity-profile operations. The adapters supply an explicit rational
polynomial ring, collision-proof internal identifiers, a fixed ordering, and a strict result encoding. Generic-fiber
degree evidence also carries a lift matrix back to the source ideal and has an
explicit independent verification pass. Singular does not define the public
request or result types.

SageMath is the current development-time differential oracle for selected
Singular-backed algorithms. It is not a Jacobian runtime dependency or a
required CI environment; the Singular adapter and bounded repository-owned
property tests carry the required evidence.

## QEPCAD

Only `real_algebraic.plane_semialgebraic.component_profile.compute` uses
[QEPCAD B 1.74](https://www.usna.edu/Users/cs/wcbrown/qepcad/B/QEPCAD.html)
as a private child-process backend for exact connected components of bounded-size
sign tables in two variables. It returns one exact representative per component
and assigns supplied points to those components. Univariate root isolation,
Sturm operations, and ordinary polynomial arithmetic do not depend on QEPCAD.
The required CI lane installs Debian's pinned
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

### Maintenance and replacement

QEPCAD is a legacy dependency, not an actively developed backend claim. As of
2026-09-06, the latest commit on its upstream default branch is
[0f570797, dated 2021-03-09](https://github.com/chriswestbrown/qepcad/commit/0f57079731afb850d0960f2265f00de8b9e213a0).
The version pin identifies the protocol tested by this adapter; it is not
evidence of upstream maintenance. Missing QEPCAD remains an explicit execution
failure for nondegenerate requests, rather than an approximate topology result.

Replacement must supply complete connectivity, including lower-dimensional
cells, strict inequalities, singular contacts, unbounded components, and exact
sample classification. A satisfiability witness or a CAD without the necessary
incidence information does not establish that postcondition.

| Candidate | Fit and remaining work |
| --- | --- |
| [CGAL algebraic-curve arrangements](https://doc.cgal.org/latest/Arrangement_on_surface_2/classCGAL_1_1Arr__algebraic__segment__traits__2.html) | A candidate foundation for a replacement: exact plane-curve geometry and arrangement incidence. Requires a C++ adapter, sign evaluation on all cell dimensions, connectivity extraction, exact representative conversion, and new admission bounds. This is an architectural assessment, not a validated replacement. |
| [Tarski](https://github.com/chriswestbrown/tarski) | Its source includes and builds QEPCAD and SACLIB. Switching to Tarski would not remove the underlying dependency. |
| [Z3 NLSat](https://z3prover.github.io/papers/programmingz3.html#sec-nlsat) | Provides nonlinear-real satisfiability machinery. Its solver result alone does not supply this operation's complete component partition. |
| [Wolfram SemialgebraicComponentInstances](https://reference.wolfram.com/language/ref/SemialgebraicComponentInstances.html) | Supplies component instances and is a candidate independent reference. Integrating it requires a separately provisioned runtime and evaluation of representative and sample-classification semantics. |

Retain the existing exact plane-component regressions when evaluating a
replacement: annulus complements, disjoint disks, strict versus closed
boundaries, touching lemniscate lobes, degree-sixteen intersection coordinates,
and invariance under reordered atoms and added samples. Do not remove QEPCAD
until a replacement satisfies the same accepted mathematical contract.

## Runtime availability and installation

System-runtime requirements belong to operation declarations as
`runtime_requirements`. Environment diagnostics belong to
`jacobian.backends.check_backend` and MCP inspection, outside mathematical
values. Adapters translate unavailable runtimes to `BackendUnavailableError`;
transport presents the recovery hint without changing result schemas.

Declare every system runtime an operation can require, including calls through
another domain's adapter. Native shortcuts do not remove that declaration: a
projective line can be handled without Singular, while degree-two and degree-three
singularity profiles need it. Declarations describe possible execution
requirements; they neither probe the environment nor guarantee availability.

When adding a backend consumer, cover both installed-backend mathematical
behavior and absent-backend recovery through the native and MCP entry points.
Preserve `BackendUnavailableError` through intermediate domain error handling.
Update the requirement list in the installation guide and the source/test
selection described in the [testing strategy](testing-strategy.md#ci-lifecycle).
See [backend requirements](../how-to/backend-requirements.md) for the default
Python dependencies, optional system installations, and supported versions.
