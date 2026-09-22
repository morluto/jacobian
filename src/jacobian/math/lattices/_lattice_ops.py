"""Exact integer-lattice structural kernels.

These functions are pure computation kernels backed by Python-FLINT for
integer rank, determinant, Smith and Hermite normal forms, plus exact rational
nullspace operations.  They accept plain integer lists-of-lists and return
plain Python values (ints, lists-of-lists of ints, lists-of-lists of
``Fraction``).  The public ``_models`` layer converts the returned values into
canonical wire types.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal

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


def integer_rank(entries: list[list[int]]) -> int:
    """Return the exact rank over ``QQ`` of an integer entry matrix."""
    from flint import fmpz_mat

    return int(fmpz_mat(entries).rank())


def integer_determinant(entries: list[list[int]]) -> int:
    from flint import fmpz_mat

    return int(fmpz_mat(entries).det())


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
    if not entries:
        return []
    from jacobian.math.matrices._flint import integer_smith_normal_form

    snf = integer_smith_normal_form(
        tuple(tuple(int(value) for value in row) for row in entries)
    )
    factors: list[int] = []
    for index in range(min(len(snf), len(snf[0]))):
        value = snf[index][index]
        if value != 0:
            factors.append(abs(int(value)))
    return factors


# ---------------------------------------------------------------------------
# Dual, saturation, discriminant group
# ---------------------------------------------------------------------------


def dual_basis(entries: list[list[int]]) -> list[list[Fraction]]:
    """Return a rational dual basis of the lattice spanned by ``entries``.

    For a full-row-rank basis ``B`` of rank ``r``, the row dual basis is
    ``B^* = (B B^T)^{-1} B`` so that
    ``B^* B^T = I_r``.
    """
    if not entries:
        return []
    from flint import fmpq_mat, fmpz_mat

    basis = fmpz_mat(entries)
    gram = basis * basis.transpose()
    # Solve G X = B directly: this avoids materializing G^{-1} and then
    # multiplying, while retaining exact rational arithmetic.
    dual = fmpq_mat(gram).solve(fmpq_mat(basis))
    return [
        [
            Fraction(int(dual[row, column].p), int(dual[row, column].q))
            for column in range(dual.ncols())
        ]
        for row in range(dual.nrows())
    ]


def saturate_lattice(
    entries: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], int]:
    """Return ``(saturated_basis, inclusion, index)`` for an admitted lattice.

    The caller establishes full row rank before entering this private kernel.
    ``saturated_basis`` spans ``sat(L) = span_Q(L) cap ZZ^n`` in HNF canonical
    form.  ``inclusion`` is the integer matrix ``C`` with ``B = C @ sat``.
    ``index`` is the finite index ``[sat(L) : L]``.
    """
    if not entries:
        return [], [], 1

    from sympy import ZZ, Matrix
    from sympy.matrices.normalforms import hermite_normal_form
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    basis = Matrix(entries)
    rows = basis.rows
    # If S B T = D is a Smith decomposition, the first r rows of T^{-1}
    # form a primitive integer basis of span_Q(B) cap ZZ^n.  Row HNF makes
    # that basis canonical without changing its lattice.
    domain_basis = DomainMatrix.from_Matrix(basis).convert_to(ZZ)
    _, _, right = smith_normal_decomp(domain_basis)
    primitive_rows: Any = right.to_Matrix().inv()[:rows, :]
    sat: Any = hermite_normal_form(primitive_rows.T).T

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

    # B = C @ sat identifies L with the full-rank row lattice of the
    # square integer matrix C in ZZ^r, so [sat(L) : L] = |det C|.
    index = abs(integer_determinant(inclusion_rows))

    return (
        _sympy_to_int_list(sat),
        inclusion_rows,
        index,
    )


def sublattice_index(
    embedding: list[list[int]],
    parent_rank: int,
) -> tuple[int | Literal["INFINITE"], list[int], int]:
    """Return ``(index, torsion_invariant_factors, free_rank)`` for an inclusion.

    ``embedding`` is the integer matrix ``E`` with ``sublattice = E @ parent``.
    The quotient ``parent / sublattice`` is computed from the Smith normal form
    of ``E``. Positive free rank means infinite index, independently of torsion.
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
    return "INFINITE" if free_rank else index, finite, free_rank


def discriminant_group(entries: list[list[int]]) -> tuple[int, list[int]]:
    """Return ``(discriminant_order, invariant_factors)``.

    The discriminant group ``L^*/L`` has order ``|det G|`` where ``G`` is the
    Gram matrix.  Its invariant factors are the Smith invariant factors of
    ``G``.
    """
    gram = gram_matrix(entries)
    det = abs(integer_determinant(gram))
    if det == 0:
        raise ValueError("discriminant group requires a nondegenerate lattice")
    factors = smith_invariant_factors(gram)
    # Only the nonzero factors matter for the group structure.
    finite = [f for f in factors if f != 0 and f != 1]
    return det, finite


def orthogonal_complement(
    entries: list[list[int]], *, ambient_dimension: int
) -> list[list[Fraction]]:
    """Return a rational basis for the orthogonal complement in ``QQ^n``.

    The orthogonal complement of the row space of ``B`` is the right nullspace
    of ``B`` (vectors ``x`` with ``B x = 0``).
    """
    if not entries:
        return [
            [Fraction(int(i == j)) for j in range(ambient_dimension)]
            for i in range(ambient_dimension)
        ]

    from flint import fmpz_mat

    # FLINT returns a column basis K with B K = 0. Its integer vectors span
    # the same rational nullspace as a rational basis and compose directly
    # with the canonical rational result type.
    kernel, nullity = fmpz_mat(entries).nullspace()
    return [
        [Fraction(int(kernel[column, basis])) for column in range(ambient_dimension)]
        for basis in range(int(nullity))
    ]


def direct_sum(
    first: list[list[int]],
    second: list[list[int]],
    *,
    first_dimension: int,
    second_dimension: int,
) -> list[list[int]]:
    """Return the block-diagonal direct-sum basis."""
    rows_first = len(first)
    cols_first = first_dimension
    rows_second = len(second)
    cols_second = second_dimension
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


def orthogonal_sum(
    first: list[list[int]],
    second: list[list[int]],
    *,
    first_dimension: int,
    second_dimension: int,
) -> list[list[int]]:
    """Return the orthogonal-sum basis (block-diagonal under standard form).

    For the standard bilinear form the orthogonal sum embedding is the same
    block-diagonal construction as the direct sum; the distinction is semantic
    and recorded in the result relation.
    """
    return direct_sum(
        first,
        second,
        first_dimension=first_dimension,
        second_dimension=second_dimension,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sympy_to_int_list(matrix: Any) -> list[list[int]]:
    rows, cols = matrix.rows, matrix.cols
    return [[int(matrix[i, j]) for j in range(cols)] for i in range(rows)]
