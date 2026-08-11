# Native Python API

Jacobian provides a deliberately small native-value API for deterministic
mathematical computations. It is independent of the MCP transport and does not
construct a capability runtime.

```python
from fractions import Fraction

import networkx as nx
import sympy

from jacobian.math import (
    arithmetic,
    graphs,
    matrices,
    polynomials,
    prime_field_linear_algebra,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

half = arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6))
matrix = sympy.Matrix([[1, 2], [3, 4]])
determinant = matrices.determinant(matrix)
inverse = matrices.inverse(matrix)
triangles = graphs.triangle_count(nx.cycle_graph(3))
derivative = polynomials.derivative(sympy.Poly(sympy.Symbol("x") ** 2, sympy.Symbol("x")))
binary_rank = prime_field_linear_algebra.rank(
    PrimeFieldMatrix(prime=2, entries=((1, 0), (1, 1)), columns=2)
)
```

The supported modules and symbols are:

- `jacobian.math.arithmetic`: `absolute_value`, `sign`, `reciprocal`,
  `sum_rationals`, and `quotient`;
- `jacobian.math.matrices`: `determinant`, `rank`, `rref`, `inverse`, and
  `trace`;
- `jacobian.math.graphs`: `triangle_count`, `diameter`, and `is_eulerian`;
- `jacobian.math.polynomials`: `derivative`, `gcdex`, and `resultant`; and
- `jacobian.math.prime_field_linear_algebra`: `PrimeFieldMatrix`, `rank`,
  `rref`, `nullspace`, `column_basis`, and `quotient_basis`.

Arithmetic functions return Python `int` or `fractions.Fraction` values. Matrix
functions accept and return SymPy matrices and exact SymPy scalar values. Graph
functions accept undirected simple NetworkX `Graph` objects. Polynomial
functions accept and return exact SymPy `Poly` values or their exact scalar
results. Each module's
`__all__` is the authoritative public symbol manifest; other implementation
modules remain internal.

This API is the authoritative mathematical implementation rather than a facade
over `math.run`. Installed operations parse their typed request once, convert
once to the documented semantic input, call the same public function, and
serialize once. Runtime, catalog, storage, publication, provider installation,
MCP, and checker-authority objects are not part of this namespace.

Each public function has one canonical semantic input type. A maintained
backend type is public only when it already carries complete semantics;
otherwise the owning `jacobian.math.<domain>` package provides an immutable
value and explicit interoperability constructor. Provider-specific transient
objects never become wire, artifact, or cross-provider composition identity.
