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

The native surface also retains useful deterministic helpers intentionally
excluded from `math.find`, including classical combinatorial numbers, basic
formal-series transformations, Young-diagram projections, graph transforms and
decomposition projections, DFA complement, continued-fraction convergents, and
finite-metric balls. Their absence from the public operation catalog is
deliberate: native availability does not create a distinct agent discovery
intent.
