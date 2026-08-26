"""Exact Smith normal decomposition used by trusted producers.

The canonical Smith diagonal and divisibility chain are mathematical
invariants.  The unimodular transformations ``U`` and ``V`` and every
representative derived from them are deterministic for the pinned SymPy
version but are **not** byte-identical across backend versions: compatibility
is semantic (``D = U A V``, unimodularity, canonical diagonal) rather than
representational.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.matrices.certified_snf.values import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)

Matrix = list[list[int]]


@dataclass(frozen=True, slots=True)
class SmithReduction:
    source: Matrix
    diagonal: Matrix
    left: Matrix
    right: Matrix
    rank: int
    invariant_factors: tuple[int, ...]
    left_determinant: int
    right_determinant: int


def zero_matrix(rows: int, columns: int) -> Matrix:
    return [[0 for _ in range(columns)] for _ in range(rows)]


def identity_matrix(size: int) -> Matrix:
    result = zero_matrix(size, size)
    for index in range(size):
        result[index][index] = 1
    return result


def matrix_shape(matrix: Matrix, *, columns_if_empty: int = 0) -> tuple[int, int]:
    return len(matrix), len(matrix[0]) if matrix else columns_if_empty


def matrix_multiply(
    left: Matrix,
    right: Matrix,
    *,
    right_columns_if_empty: int = 0,
) -> Matrix:
    left_rows = len(left)
    middle = len(left[0]) if left else len(right)
    right_rows = len(right)
    right_columns = len(right[0]) if right else right_columns_if_empty
    if middle != right_rows:
        raise ValueError("integer matrices are not composable")
    return [
        [
            sum(left[row][index] * right[index][column] for index in range(middle))
            for column in range(right_columns)
        ]
        for row in range(left_rows)
    ]


def matrix_vector_multiply(matrix: Matrix, vector: list[int]) -> list[int]:
    if matrix and len(matrix[0]) != len(vector):
        raise ValueError("integer matrix and vector are not composable")
    return [
        sum(value * vector[index] for index, value in enumerate(row)) for row in matrix
    ]


def matrix_columns(matrix: Matrix, start: int = 0) -> Matrix:
    if not matrix:
        return []
    return [
        [matrix[row][column] for column in range(start, len(matrix[0]))]
        for row in range(len(matrix))
    ]


def inverse_unimodular(matrix: Matrix) -> Matrix:
    """Invert a unimodular integer matrix with SymPy ``DomainMatrix.inv_den``."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("unimodular inverse requires a square matrix")
    if size == 0:
        return []

    from sympy import ZZ, eye
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    domain = DomainMatrix.from_list_sympy(size, size, matrix).convert_to(ZZ)
    try:
        numerator, denominator = domain.inv_den()
    except DMNonInvertibleMatrixError as exc:
        raise ValueError("matrix is singular") from exc
    numerator, denominator = numerator.cancel_denom(denominator)
    if denominator != ZZ.one:
        raise ValueError("matrix is not unimodular")
    if domain.matmul(numerator).to_Matrix() != eye(size):
        raise ArithmeticError("unimodular inverse does not recover the identity")
    return [[int(value) for value in row] for row in numerator.to_Matrix().tolist()]


def matrix_determinant(matrix: Matrix) -> int:
    """Exact determinant of one square integer matrix with SymPy over ``ZZ``."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square integer matrix")
    if size == 0:
        return 1

    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix

    domain_matrix = DomainMatrix(
        [[ZZ(value) for value in row] for row in matrix],
        (size, size),
        ZZ,
    )
    return int(domain_matrix.det())


def smith_reduce(
    source: Matrix,
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> SmithReduction:
    """Return a canonical Smith diagonal and explicit unimodular transformations.

    Delegates the decomposition to SymPy's ``smith_normal_decomp`` over ``ZZ``
    and converts the result to native integer matrices.  The canonical diagonal
    and invariant factors are mathematical invariants; ``U`` and ``V`` are
    deterministic for the pinned SymPy version but may differ from other
    backends.  Fail-closed checks verify ``D = U A V``, unimodularity, and the
    positive divisibility chain before returning.
    """

    import sympy
    from sympy.matrices.normalforms import smith_normal_decomp

    rows = len(source) if row_count is None else row_count
    columns = (
        (len(source[0]) if source else 0) if column_count is None else column_count
    )
    if len(source) != rows or any(len(row) != columns for row in source):
        raise ValueError("source entries do not match the declared matrix shape")
    original = [row[:] for row in source]

    # smith_normal_decomp accepts a plain SymPy Matrix and handles every shape,
    # including 0xm and nx0 matrices, returning identity transformations for
    # the empty side.
    if rows and columns:
        sympy_source = sympy.Matrix([[int(value) for value in row] for row in original])
    else:
        sympy_source = sympy.Matrix(rows, columns, [])

    diagonal, left, right = smith_normal_decomp(sympy_source, domain=sympy.ZZ)

    diagonal_matrix = [
        [int(diagonal[row, column]) for column in range(columns)] for row in range(rows)
    ]
    left_matrix = [
        [int(left[row, column]) for column in range(rows)] for row in range(rows)
    ]
    right_matrix = [
        [int(right[row, column]) for column in range(columns)] for row in range(columns)
    ]

    diagonal_count = min(rows, columns)
    factors = tuple(
        diagonal_matrix[index][index]
        for index in range(diagonal_count)
        if diagonal_matrix[index][index] != 0
    )
    if any(value <= 0 for value in factors) or any(
        right_factor % left_factor for left_factor, right_factor in pairwise(factors)
    ):
        raise ArithmeticError("Smith reduction did not produce a canonical diagonal")
    if (
        matrix_multiply(matrix_multiply(left_matrix, original), right_matrix)
        != diagonal_matrix
    ):
        raise ArithmeticError("Smith transformations do not bind the source")
    left_determinant = int(left.det())
    right_determinant = int(right.det())
    if abs(left_determinant) != 1 or abs(right_determinant) != 1:
        raise ArithmeticError("Smith transformations are not unimodular")
    return SmithReduction(
        source=original,
        diagonal=diagonal_matrix,
        left=left_matrix,
        right=right_matrix,
        rank=len(factors),
        invariant_factors=factors,
        left_determinant=left_determinant,
        right_determinant=right_determinant,
    )


def _contract_matrix(
    entries: Matrix,
    *,
    rows: int,
    columns: int,
) -> CertifiedIntegerMatrix:
    return CertifiedIntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(
            tuple(format_canonical_integer(value) for value in row) for row in entries
        ),
    )


def certificate_from_reduction(
    reduction: SmithReduction,
) -> SmithNormalFormCertificate:
    rows = len(reduction.source)
    columns = len(reduction.source[0]) if reduction.source else len(reduction.right)
    left_determinant: Literal["-1", "1"] = (
        "1" if reduction.left_determinant == 1 else "-1"
    )
    right_determinant: Literal["-1", "1"] = (
        "1" if reduction.right_determinant == 1 else "-1"
    )
    return SmithNormalFormCertificate._from_kernel(
        source=_contract_matrix(reduction.source, rows=rows, columns=columns),
        diagonal=_contract_matrix(reduction.diagonal, rows=rows, columns=columns),
        left_transformation=_contract_matrix(
            reduction.left,
            rows=rows,
            columns=rows,
        ),
        right_transformation=_contract_matrix(
            reduction.right,
            rows=columns,
            columns=columns,
        ),
        rank=reduction.rank,
        invariant_factors=tuple(
            format_canonical_integer(value) for value in reduction.invariant_factors
        ),
        left_determinant=left_determinant,
        right_determinant=right_determinant,
    )


def verify_smith_normal_form_certificate(
    certificate: SmithNormalFormCertificate,
) -> bool:
    """Verify an independently supplied certificate in the admitted envelope.

    Parsing the certificate only establishes its canonical shape and Smith
    diagonal.  This explicit path checks the claimed transformations and
    determinant signs after first requiring the source to fit the public
    certified-Smith request envelope, which bounds the exact replay.
    """

    from pydantic import ValidationError

    from jacobian.math.matrices.certified_snf._models import (
        CertifiedSmithNormalFormRequest,
    )

    try:
        CertifiedSmithNormalFormRequest(matrix=certificate.source)
    except ValidationError:
        return False

    source = [
        [parse_canonical_integer(value) for value in row]
        for row in certificate.source.entries
    ]
    diagonal = [
        [parse_canonical_integer(value) for value in row]
        for row in certificate.diagonal.entries
    ]
    left = [
        [parse_canonical_integer(value) for value in row]
        for row in certificate.left_transformation.entries
    ]
    right = [
        [parse_canonical_integer(value) for value in row]
        for row in certificate.right_transformation.entries
    ]
    try:
        if matrix_multiply(matrix_multiply(left, source), right) != diagonal:
            return False
        return all(
            matrix_determinant(transformation) == int(determinant)
            for transformation, determinant in (
                (left, certificate.left_determinant),
                (right, certificate.right_determinant),
            )
        )
    except (ArithmeticError, ValueError):
        return False


__all__ = [
    "Matrix",
    "SmithReduction",
    "certificate_from_reduction",
    "identity_matrix",
    "inverse_unimodular",
    "matrix_columns",
    "matrix_determinant",
    "matrix_multiply",
    "matrix_shape",
    "matrix_vector_multiply",
    "smith_reduce",
    "verify_smith_normal_form_certificate",
    "zero_matrix",
]
