"""Domain functions for finite-dimensional algebra operations."""

from __future__ import annotations

from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    CenterResult,
)


def _nullspace_mod_p(rows: list[list[int]], n: int, p: int) -> list[list[int]]:
    """Return a basis for the nullspace of ``rows`` over ``F_p``.

    ``rows`` is an ``m x n`` matrix (list of ``m`` rows each of length ``n``).
    The nullspace basis vectors are returned in reduced row-echelon form so the
    answer is canonical for a given input.
    """
    m = len(rows)
    aug: list[list[int]] = [list(r) for r in rows]

    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break
        pivot = -1
        for r in range(pivot_row, m):
            if aug[r][col] % p != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
        inv = pow(aug[pivot_row][col] % p, -1, p)
        aug[pivot_row] = [(v * inv) % p for v in aug[pivot_row]]
        for r in range(m):
            if r != pivot_row and aug[r][col] % p != 0:
                factor = aug[r][col] % p
                aug[r] = [
                    (aug[r][k] - factor * aug[pivot_row][k]) % p for k in range(n)
                ]
        pivot_cols.append(col)
        pivot_row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * n
        vec[fc] = 1
        for ri, pc in enumerate(pivot_cols):
            vec[pc] = (-aug[ri][fc]) % p
        basis.append(vec)
    return basis


def compute_center(request: CenterRequest) -> CenterResult:
    """Compute the center of a finite-dimensional algebra over a prime field.

    The center is the nullspace of the commutator map ``[z, -]``: for each
    basis element ``e_a`` the commutator ``[z, e_a] = z*e_a - e_a*z`` vanishes
    in every coordinate ``k``, giving linear equations in the coordinates of
    ``z``.  The center is computed exactly via Gaussian elimination over
    ``F_q`` rather than by enumerating all ``q^n`` vectors.
    """
    algebra = request.algebra
    n = algebra.dimension
    q = algebra.field_order
    mult = algebra.multiplication

    rows: list[list[int]] = []
    for a in range(n):
        for k in range(n):
            row = [(mult[j][a][k] - mult[a][j][k]) % q for j in range(n)]
            rows.append(row)

    basis = _nullspace_mod_p(rows, n, q)
    center_basis = tuple(tuple(vec) for vec in basis)
    return CenterResult(
        center_basis=center_basis,
        dimension=n,
        center_dimension=len(center_basis),
    )
