"""Exact integer-lattice structural kernels.

These functions are pure computation kernels backed by SymPy (for rank,
determinant, Smith normal form, and rational nullspace) and Python-FLINT (for
Hermite normal form).  They accept plain integer lists-of-lists and return
plain Python values (ints, lists-of-lists of ints, lists-of-lists of
``Fraction``).  The public ``_models`` layer converts the returned values into
canonical wire types.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.canonical import parse_canonical_integer


def _to_int(value: str | int) -> int:
    """Coerce a canonical-string or plain int into a Python int."""
    if isinstance(value, int):
        return value
    return parse_canonical_integer(value)


__all__ = [
    "direct_sum",
    "discriminant_group",
    "dual_basis",
    "gram_matrix",
    "hermite_basis",
    "integer_determinant",
    "integer_rank",
    "orthogonal_complement",
    "orthogonal_sum",
    "saturate_lattice",
    "smith_invariant_factors",
    "sublattice_index",
]


# ---------------------------------------------------------------------------
# SymPy / FLINT helpers
# ---------------------------------------------------------------------------


def _sympy_integer_matrix(entries: list[list[int]]) -> Any:
    from sympy import Matrix

    return Matrix(entries)


def _entries_to_int(matrix: Any) -> list[list[int]]:
    rows = matrix.rows if hasattr(matrix, "rows") else len(matrix)
    cols = matrix.cols if hasattr(matrix, "cols") else len(matrix[0])
    return [[int(matrix[i, j]) for j in range(cols)] for i in range(rows)]


def integer_rank(entries: list[list[int]]) -> int:
    """Return the exact rank over ``QQ`` of an integer entry matrix."""
    from sympy import Matrix

    parsed = [[_to_int(v) for v in row] for row in entries]
    return int(Matrix(parsed).rank())


def integer_determinant(entries: list[list[int]]) -> int:
    from sympy import Matrix

    return int(Matrix(entries).det())


def gram_matrix(entries: list[list[int]]) -> list[list[int]]:
    """Return ``G = B @ B^T`` for integer basis rows ``B``."""
    rows = len(entries)
    inner_dim = len(entries[0]) if rows else 0
    gram = [[0] * rows for _ in range(rows)]
    for i in range(rows):
        for j in range(rows):
            gram[i][j] = sum(entries[i][k] * entries[j][k] for k in range(inner_dim))
    return gram


# ---------------------------------------------------------------------------
# Hermite / Smith normal forms
# ---------------------------------------------------------------------------


def hermite_basis(entries: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Return the row HNF basis ``H`` and unimodular ``U`` with ``H = U B``."""
    import flint

    source = flint.fmpz_mat(entries)
    hnf, transform = source.hnf(True)
    rows = hnf.nrows()
    cols = hnf.ncols()
    hnf_list = [[int(hnf[i, j]) for j in range(cols)] for i in range(rows)]
    t_rows = transform.nrows()
    t_cols = transform.ncols()
    transform_list = [
        [int(transform[i, j]) for j in range(t_cols)] for i in range(t_rows)
    ]
    return hnf_list, transform_list


def smith_invariant_factors(entries: list[list[int]]) -> list[int]:
    """Return the nonzero invariant factors of the integer Smith normal form."""
    from sympy import Matrix
    from sympy.matrices.normalforms import smith_normal_form

    matrix = Matrix(entries)
    snf = smith_normal_form(matrix)
    factors: list[int] = []
    for index in range(min(snf.rows, snf.cols)):
        value = snf[index, index]
        if value != 0:
            factors.append(abs(int(value)))
    return factors


# ---------------------------------------------------------------------------
# Dual, saturation, discriminant group
# ---------------------------------------------------------------------------


def _sympy_to_fractions(matrix: Any) -> list[list[Fraction]]:
    rows, cols = matrix.rows, matrix.cols
    return [
        [Fraction(int(matrix[i, j].p), int(matrix[i, j].q)) for j in range(cols)]
        for i in range(rows)
    ]


def dual_basis(entries: list[list[int]]) -> list[list[Fraction]]:
    """Return a rational dual basis of the lattice spanned by ``entries``.

    For a full-row-rank basis ``B`` of rank ``r``, the dual basis is
    ``L^* = B^T (B B^T)^{-1}``, i.e. ``B^* = B (B B^T)^{-1}`` so that
    ``B^* B^T = I_r``.
    """
    from sympy import Matrix, Rational

    basis = Matrix(entries)
    gram = basis * basis.T
    inverse = gram.inv()
    dual = basis * inverse
    rows, cols = dual.rows, dual.cols
    result: list[list[Fraction]] = []
    for i in range(rows):
        row: list[Fraction] = []
        for j in range(cols):
            entry = dual[i, j]
            if isinstance(entry, Rational):
                row.append(Fraction(int(entry.p), int(entry.q)))
            elif isinstance(entry, int):
                row.append(Fraction(entry, 1))
            else:
                row.append(Fraction(int(entry.p), int(entry.q)))
        result.append(row)
    return result


def saturate_lattice(
    entries: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], int]:
    """Return ``(saturated_basis, inclusion, index)`` for the lattice.

    ``saturated_basis`` spans ``sat(L) = span_Q(L) cap ZZ^n`` in HNF canonical
    form.  ``inclusion`` is the integer matrix ``C`` with ``sat = C @ B``.
    ``index`` is the finite index ``[sat(L) : L]``.
    """
    from math import gcd

    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    basis = Matrix(entries)
    rows = basis.rows
    if basis.rank() != rows:
        raise ValueError("lattice basis must be full row rank for saturation")

    # Saturation = ZZ^n cap span_Q(rows of B) = integer kernel of the left
    # nullspace of B.
    nullvecs = basis.T.nullspace()
    sat: Matrix
    if not nullvecs:
        sat = Matrix.eye(basis.cols)
    else:
        kernel_rows = [[vec[j] for j in range(basis.cols)] for vec in nullvecs]
        rational_kernel = Matrix(kernel_rows)
        # Clear denominators to obtain an integer matrix.
        denominators = [
            rational_kernel[i, j].q if hasattr(rational_kernel[i, j], "q") else 1
            for i in range(rational_kernel.rows)
            for j in range(rational_kernel.cols)
            if rational_kernel[i, j] != 0
        ]
        lcm = 1
        for q in denominators:
            lcm = lcm * q // gcd(lcm, q)
        integer_kernel = rational_kernel * lcm
        sat = hermite_normal_form(integer_kernel)

    # Inclusion L -> sat(L): each basis row of L is an integer combination
    # of the sat basis rows.  Solve basis = C @ sat_basis for the r x r
    # integer matrix C.
    sat_gram = sat * sat.T
    sat_gram_inv = sat_gram.inv()
    rational_inclusion = basis * sat.T * sat_gram_inv
    inclusion_rows: list[list[int]] = []
    for i in range(rational_inclusion.rows):
        row: list[int] = []
        for j in range(rational_inclusion.cols):
            value = rational_inclusion[i, j]
            if hasattr(value, "p") and hasattr(value, "q"):
                frac = Fraction(int(value.p), int(value.q))
            else:
                frac = Fraction(int(value))
            if frac.denominator != 1:
                raise ValueError(
                    "basis rows are not integer combinations of the saturated basis"
                )
            row.append(int(frac.numerator))
        inclusion_rows.append(row)

    # Index = gcd of all r x r minors of B.
    from itertools import combinations

    rank = rows
    if basis.cols < rank:
        index = 1
    else:
        minors_gcd = 0
        for cols in combinations(range(basis.cols), rank):
            minor = int(basis[:, list(cols)].det())
            minors_gcd = gcd(minors_gcd, abs(minor))
        index = minors_gcd if minors_gcd > 0 else 1

    return (
        _sympy_to_int_list(sat),
        inclusion_rows,
        index,
    )


def sublattice_index(
    embedding: list[list[int]],
    parent_rank: int,
) -> tuple[int, list[int], int]:
    """Return ``(index, invariant_factors, free_rank)`` for an inclusion.

    ``embedding`` is the integer matrix ``E`` with ``sublattice = E @ parent``.
    The quotient ``parent / sublattice`` is computed from the Smith normal form
    of ``E``.
    """
    factors = smith_invariant_factors(embedding)
    # The quotient Z^parent_rank / <rows of E> is a direct sum of the cyclic
    # groups given by the invariant factors (finite torsion) plus a free part
    # of rank parent_rank - len(factors).
    finite = [f for f in factors if f != 0 and f != 1]
    index = 1
    for f in finite:
        index *= f
    free_rank = parent_rank - len(factors)
    return index, finite, free_rank


def discriminant_group(entries: list[list[int]]) -> tuple[int, list[int]]:
    """Return ``(discriminant_order, invariant_factors)``.

    The discriminant group ``L^*/L`` has order ``|det G|`` where ``G`` is the
    Gram matrix.  Its invariant factors are the Smith invariant factors of
    ``G``.
    """
    gram = gram_matrix(entries)
    gram_sympy = _sympy_integer_matrix(gram)
    det = abs(int(gram_sympy.det()))
    if det == 0:
        raise ValueError("discriminant group requires a nondegenerate lattice")
    factors = smith_invariant_factors(gram)
    # Only the nonzero factors matter for the group structure.
    finite = [f for f in factors if f != 0 and f != 1]
    return det, finite


def orthogonal_complement(entries: list[list[int]]) -> list[list[Fraction]]:
    """Return a rational basis for the orthogonal complement in ``QQ^n``.

    The orthogonal complement of the row space of ``B`` is the right nullspace
    of ``B`` (vectors ``x`` with ``B x = 0``).
    """
    from sympy import Matrix

    basis = Matrix(entries)
    nullspace = basis.nullspace()
    if not nullspace:
        return []
    rows: list[list[Fraction]] = []
    for vec in nullspace:
        row: list[Fraction] = []
        for j in range(vec.rows):
            entry = vec[j]
            if hasattr(entry, "p") and hasattr(entry, "q"):
                row.append(Fraction(int(entry.p), int(entry.q)))
            else:
                row.append(Fraction(int(entry), 1))
        rows.append(row)
    return rows


def direct_sum(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    """Return the block-diagonal direct-sum basis."""
    rows_first = len(first)
    cols_first = len(first[0]) if rows_first else 0
    rows_second = len(second)
    cols_second = len(second[0]) if rows_second else 0
    result: list[list[int]] = []
    for i in range(rows_first):
        row = [0] * cols_first + [0] * cols_second
        for j in range(cols_first):
            row[j] = first[i][j]
        result.append(row)
    for i in range(rows_second):
        row = [0] * cols_first + [0] * cols_second
        for j in range(cols_second):
            row[cols_first + j] = second[i][j]
        result.append(row)
    return result


def orthogonal_sum(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    """Return the orthogonal-sum basis (block-diagonal under standard form).

    For the standard bilinear form the orthogonal sum embedding is the same
    block-diagonal construction as the direct sum; the distinction is semantic
    and recorded in the result relation.
    """
    return direct_sum(first, second)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hermite_form_row(basis: Any) -> Any:
    """Return the row Hermite normal form of a SymPy integer matrix."""
    from sympy.matrices.normalforms import hermite_normal_form

    return hermite_normal_form(basis, D=None)


def _sympy_to_int_list(matrix: Any) -> list[list[int]]:
    rows, cols = matrix.rows, matrix.cols
    return [[int(matrix[i, j]) for j in range(cols)] for i in range(rows)]
