"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

__all__ = [
    "adjacency_spectrum",
    "laplacian_spectrum",
    "characteristic_polynomial",
]


def _adjacency_matrix(vertex_count, edges):  # type: ignore[no-untyped-def]
    import sympy

    mat = sympy.zeros(vertex_count)
    for u, v in edges:
        mat[u, v] = 1
        mat[v, u] = 1
    return mat


def adjacency_spectrum(vertex_count, edges):  # type: ignore[no-untyped-def]
    if not 1 <= vertex_count <= 32:
        raise ValueError("graph vertex count must be between 1 and 32")
    mat = _adjacency_matrix(vertex_count, edges)  # type: ignore[no-untyped-call]
    eigenvals = mat.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def laplacian_spectrum(vertex_count, edges):  # type: ignore[no-untyped-def]
    import sympy

    if not 1 <= vertex_count <= 32:
        raise ValueError("graph vertex count must be between 1 and 32")
    adj = _adjacency_matrix(vertex_count, edges)  # type: ignore[no-untyped-call]
    degree = sympy.diag(*(sum(adj[vertex, :]) for vertex in range(vertex_count)))
    lap = degree - adj
    eigenvals = lap.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def characteristic_polynomial(  # type: ignore[no-untyped-def]
    vertex_count, edges, matrix="ADJACENCY"
):
    """Compute the monic characteristic polynomial of a graph matrix.

    Returns ``(degree, coefficients_descending)`` where the coefficients are
    canonical decimal strings and the polynomial is ``det(lambda*I - M)``.
    """
    import sympy

    if not 1 <= vertex_count <= 32:
        raise ValueError("graph vertex count must be between 1 and 32")
    mat = _adjacency_matrix(vertex_count, edges)  # type: ignore[no-untyped-call]
    if matrix == "LAPLACIAN":
        degree = sympy.diag(*(sum(mat[vertex, :]) for vertex in range(vertex_count)))
        mat = degree - mat
    elif matrix != "ADJACENCY":
        raise ValueError("matrix must be ADJACENCY or LAPLACIAN")
    lam = sympy.Symbol("lambda")
    poly = (sympy.eye(vertex_count) * lam - mat).det()
    coeffs = [str(c) for c in sympy.Poly(poly, lam).all_coeffs()]
    return len(coeffs) - 1, tuple(coeffs)
