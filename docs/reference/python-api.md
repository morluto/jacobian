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
    finite_fields,
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
field = finite_fields.finite_field(2, (1, 1, 0, 1))
a = finite_fields.element(field, (0, 1, 0))
```

The supported modules and symbols are:

- `jacobian.math.arithmetic`: `absolute_value`, `sign`, `reciprocal`,
  `sum_rationals`, and `quotient`;
- `jacobian.math.matrices`: `determinant`, `rank`, `rref`, `inverse`, and
  `trace`;
- `jacobian.math.graphs`: `SimpleUndirectedGraph`, `GraphCompositionInput`,
  `explicit_graph`, `compose_graphs`, `triangle_count`, `diameter`, and
  `is_eulerian`;
- `jacobian.math.polynomials`: `derivative`, `discriminant`, `divide`,
  `evaluate`, `factorization`, `gcdex`, `groebner_basis`, `integral`,
  `partial_fractions`, `resultant`, and `square_free_decomposition`;
- `jacobian.math.prime_field_linear_algebra`: `PrimeFieldMatrix`, `rank`,
  `rref`, `nullspace`, `column_basis`, and `quotient_basis`; and
- `jacobian.math.finite_fields`: exact presentation-, parent-, and axis-bound
  values plus projective normalization, projective-line enumeration, explicit
  restriction of scalars, direction-bound rank ledgers, orbit aggregation,
  finite polynomial maps, complete tables, fibers, and bound collision and
  permutation certificates.

`SimpleUndirectedGraph` is owned by `jacobian.math.graphs`; graph operation and
artifact boundaries convert it explicitly to their wire contract. Native
callers therefore do not depend on a capability-specific contract module.

`projective_line` returns a `ProjectiveLine` value rather than an unbound tuple,
so its presentation, axis, completeness, order, and digest remain attached.

Arithmetic functions return Python `int` or `fractions.Fraction` values. Matrix
functions accept and return SymPy matrices and exact SymPy scalar values. Graph
algorithms accept undirected simple NetworkX `Graph` objects; graph construction
and composition return the owned immutable graph value. Polynomial functions
accept and return exact SymPy `Poly` values or their exact scalar results. Each
module's
`__all__` is the authoritative public symbol manifest; other implementation
modules remain internal.

Jacobian builds these functions on maintained mathematical libraries rather
than reimplementing their algorithms. Small private modules such as `_sympy`
and `_flint` contain lazy backend calls, exact conversions, and backend-specific
normalization. They are implementation boundaries, not public wrapper APIs or
a generic adapter framework. Public functions validate their documented
semantic contract—for example, `groebner_basis` accepts only SymPy `Poly`
values over `QQ`—and then delegate directly to the maintained backend.
In short, Jacobian wraps semantic contracts, not entire libraries.

Finite-extension values bind the exact modulus, generator, ordered power basis,
and coordinate encoding. Matrix, subspace, projective, and linear-map values
also bind their parents and ordered axes. SymPy validates presentations and
normalizes projective values. Backend conversion stays private and lazy;
importing `jacobian.math` does not probe or import backend implementations until
a corresponding function is called. The maintained backends are exact package
dependencies, but they do not become mathematical value identity.

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
