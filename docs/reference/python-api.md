# Native Python API

Jacobian exposes a small native mathematical API under `jacobian.math`. Native
functions call the domain kernels directly and are independent of MCP.

```python
from fractions import Fraction

import sympy

from jacobian.math import matrices, polynomials
from jacobian.math.number_theory import arithmetic

half = arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6))
matrix = sympy.Matrix([[1, 2], [3, 4]])
determinant = matrices.determinant(matrix)
```

Each public `jacobian.math.<domain>` module declares its supported names in
`__all__`; that is the authoritative native API. Functions accept domain values
or a maintained backend type when it already carries the complete mathematical
meaning. Private backend modules perform lazy conversions and calls to SymPy,
NetworkX, FLINT, or Z3.

Public import paths follow mathematical ownership. Large owners expose
recognizable subdivisions, for example
`jacobian.math.graphs.chip_firing` or
`jacobian.math.polynomials.interpolation`; backend and transport structure does
not appear in the public path. When a pre-stable package is folded into its
canonical owner, callers move to the owner path in the same release. The old
path is removed rather than retained as a forwarding package, and operation IDs
and wire schemas remain unchanged unless their mathematical contract changes
independently.

Native values are mathematical values rather than wire envelopes. Native and
MCP calls use the same domain-owned request admission, kernel path, canonical
result construction, and typed outcome semantics. Most native functions simply
normalize and validate their input, perform the computation, and return a
canonical value. An operation uses a distinct execution plan only when
admission derives information that its kernel or result construction needs to
reuse. The MCP path adds only wire parsing and the final transport projection;
native code must not
inherit MCP byte/depth/echo limits unless those limits are part of the
mathematical operation itself.

An MCP operation parses one wire request before calling the shared domain
admission function. Native callers call that domain function directly rather
than constructing a wire Pydantic model. Native and wire parity tests should
assert equal exact results and typed outcomes, and should document any
difference as an explicit transport-only constraint.

Pydantic request models are transport contracts, not containers for hidden
execution state. They may enforce typed shape, canonical representation, and
cheap intrinsic cross-field consistency. They must not store admission plans in
``PrivateAttr`` fields or perform backend calls, candidate enumeration,
algorithm selection, or result-work planning in validators. Those steps belong
to the native function after wire-to-domain conversion.

## Canonical native values

The [interoperability contract](value-interoperability.md) distinguishes shared
values, meaningful refinements, representation transforms, and changes of
parent. It also defines when conversions belong only in the native API and
what serialization does and does not establish.

Each mathematical value has one owner-defined public type, normally in the
domain's `values.py`. Producers return that type and consumers accept the same
type directly. Operation-specific request and result models may contain a
canonical value alongside genuine operation parameters, but must not reproduce
it as a parallel set of fields.

For example, the finite-field API follows this ownership chain:

```text
finite_field(...) -> FiniteFieldPresentation
FiniteDimensionalSubspace(...) -> FiniteDimensionalSubspace
linear_map_rank(subspace, direction) accepts that FiniteDimensionalSubspace
```

The same rule applies after serialization: a producer's canonical value must
pass through the consumer's typed boundary without the caller reconstructing
its field presentation, axes, ambient dimension, normalization, or other
mathematical context. Empty and degenerate values retain that context too.

Integer encoding does not authorize flattening that value. A dynamics request
continues to accept `RationalPolynomial` or `FinitePolynomialMap`, rather than
separate coefficient and field parameters; only their integer leaves serialize
as decimal strings. This preserves direct producer-to-consumer composition and
keeps parent, variable, and domain/codomain data available to admission.

For example, a QQ matrix with `row_count=0`, `column_count=3`, and
`entries=[]` is a map from a three-dimensional space to the zero space.
Its dimensions survive dense/sparse conversion, backend conversion, and JSON
round trips. An empty quadratic matrix additionally supplies its `radicand`;
an embedded number-field matrix retains its `embedding`. Never infer a parent
from an entry when there may be no entries.

Likewise, `FiniteDimensionalSubspace` supplies `presentation`, `row_axis`, and
`column_axis` independently of its `basis`. An empty `basis` and empty
`basis_axis.labels` represent the zero subspace in that ambient matrix space;
scalar restriction retains the target axis even when its matrix has zero
columns. Nonempty basis matrices must match the declared parent and axes.
Consumers establish the basis's independence at admission, not during parsing.

### Integer values and JSON

Migrated integer fields hold Python `int` values, not decimal text. The same
model owns both native computation and JSON encoding:

```python
from jacobian.math.matrices.values import IntegerMatrix

matrix = IntegerMatrix(entries=((2**53 + 1,),))
assert matrix.entries[0][0] == 2**53 + 1
assert matrix.model_dump()["entries"] == ((2**53 + 1,),)
assert matrix.model_dump(mode="json")["entries"] == [["9007199254740993"]]
assert IntegerMatrix.model_validate_json(matrix.model_dump_json()) == matrix
```

Use native integers in mathematical tests. Test malformed decimal strings and
serialization separately through `model_validate_json()`. Passing a decoded
JSON dictionary to `model_validate()` selects the native path and does not
decode its strings. This encoding choice does not increase an operation's
work or output limits. See the [migration requirements](value-interoperability.md#requirements-for-a-native-integer-codec)
for the remaining legacy-field and compound-rational scope.

### Finite abelian groups

The abelian operations use `AbelianPresentation(invariant_factors=(2, 12))`.
Its invariant factors and all element coordinates are exact Python integers.
`model_dump()` preserves them; `model_dump(mode="json")` and
`model_dump_json()` encode them as canonical decimal strings.
`model_validate_json()` validates and decodes that encoding back into the same
value type. Native construction does not accept numeric strings, and JSON
decoding does not accept numbers for these fields.

`reduce_element(group, coordinates)` returns an `ElementReduceResult` retaining
the source coordinates and an `AbelianElement` with canonical coordinates.
`element_order` and `elements_equal` retain their source group and canonical
elements in their result values. `generated_subgroup` and `quotient_group`
similarly return parent-bound `AbelianSubgroup` values. Different presentations
are not implicitly identified merely because they present isomorphic groups.

`normalize_presentation(source)` retains the source cyclic-factor presentation
and returns an invariant-factor `AbelianPresentation`; quotient computation
returns an abstract `AbelianQuotient`. Neither result supplies a coordinate
isomorphism or quotient projection. Group order and exponent are mathematical
result fields, not duplicated serialized claims. Work bounds are checked at
native operation admission. Reduction accepts exact Python integers natively
and canonical decimal-string coordinates through JSON, with a 32,768-digit
encoding guard in both paths.

The abelian Smith operations still retain a 4,096 group-order admission cap;
do not interpret canonical integer encoding as evidence that this separate
scale limitation has been repaired. Other finite-group and character operations
retain their own bounded coordinate contracts.
See [exact integer migrations](value-interoperability.md#exact-integers-representation-is-not-a-work-limit)
and [large-group admission](public-operation-admission.md#large-integers-and-large-finite-groups)
for the required repairs.

A carrier's structural size limit is not an operation's admission limit. The QQ
matrix carrier supports axes through 4,096 so it can represent the eventual
hitting operation's state space. Each consumer separately admits its work,
intermediate growth, and result size; this does not make arbitrary dense
4,096-by-4,096 computations admissible.

Deserialization checks structure and context, not mathematical claims such as
primality, lattice rank, an order relation, or a functional graph's cycles.
Operations check the properties they rely on during admission. A parsed result
is still a claim from its source; serialization does not turn it into an
independently verified certificate. Do not add certificates unless a consumer
needs a concrete mathematical witness or checking relation.

The native surface also retains useful deterministic helpers intentionally
excluded from `math.find`, including classical combinatorial numbers, basic
formal-series transformations, Young-diagram projections, graph transforms and
decomposition projections, DFA complement, continued-fraction convergents, and
finite-metric balls. Their absence from the public operation catalog is
deliberate: native availability does not create a distinct agent discovery
intent.

## Optional native runtimes

Python package dependencies install with Jacobian. Singular and QEPCAD are
optional system runtimes for a small declared subset of operations; ordinary
Python calls and imports do not require them. Use
`from jacobian.backends import check_backend` and `check_backend("singular")`
(or `"qepcad"`) to inspect the current environment. Calls that need an unavailable
runtime raise `jacobian.backends.BackendUnavailableError`, with `backend`,
`required_version`, `detail`, and `installation` attributes. No runtime is
automatically installed. See [backend requirements](../how-to/backend-requirements.md)
for versions, operation coverage, and deployment instructions.
