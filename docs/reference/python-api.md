# Native Python API

Jacobian provides a deliberately small native-value API for deterministic
mathematical computations. It is independent of the MCP transport and does not
construct a capability runtime.

```python
from fractions import Fraction

import networkx as nx
import sympy

from jacobian.math import arithmetic, graphs, matrices, polynomials

half = arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6))
inverse = matrices.inverse(sympy.Matrix([[1, 2], [3, 4]]))
triangles = graphs.triangle_count(nx.cycle_graph(3))
derivative = polynomials.derivative(sympy.Poly(sympy.Symbol("x") ** 2, sympy.Symbol("x")))
```

The supported modules and symbols are:

- `jacobian.math.arithmetic`: `absolute_value`, `sign`, `reciprocal`,
  `sum_rationals`, and `quotient`;
- `jacobian.math.matrices`: `rref`, `inverse`, and `trace`; and
- `jacobian.math.graphs`: `triangle_count`, `diameter`, and `is_eulerian`; and
- `jacobian.math.polynomials`: `derivative`, `gcdex`, and `resultant`.

Arithmetic functions return Python `int` or `fractions.Fraction` values. Matrix
functions accept and return SymPy matrices and exact SymPy scalar values. Graph
functions accept undirected simple NetworkX `Graph` objects. Polynomial
functions accept and return exact SymPy `Poly` values or their exact scalar
results. Each module's
`__all__` is the authoritative public symbol manifest; other implementation
modules remain internal.

This API shares typed mathematical kernels with the corresponding capability
implementations, but it is not a facade over `math.run`. Capability
requests, result contracts, artifacts, provenance, completeness, and
verification remain available through the capability runtime and retain their
existing wire semantics.
