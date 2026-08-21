# Native Python API

Jacobian exposes a small native mathematical API under `jacobian.math`. Native
functions call the domain kernels directly and are independent of MCP.

```python
from fractions import Fraction

import sympy

from jacobian.math import arithmetic, matrices, polynomials

half = arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6))
matrix = sympy.Matrix([[1, 2], [3, 4]])
determinant = matrices.determinant(matrix)
```

Each public `jacobian.math.<domain>` module declares its supported names in
`__all__`; that is the authoritative native API. Functions accept domain values
or a maintained backend type when it already carries the complete mathematical
meaning. Private backend modules perform lazy conversions and calls to SymPy,
NetworkX, FLINT, or Z3.

Native values are mathematical values rather than wire envelopes. An operation
parses one typed request, calls the same domain kernel, and serializes one typed
result at the final MCP boundary.

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
