"""Exact symbolic matrix operations backed by SymPy.

The entries are canonical rational-function values over a declared variable
axis.  SymPy objects are constructed programmatically from validated sparse
terms; this module never parses caller text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic_core import PydanticCustomError
from sympy.matrices.exceptions import MatrixError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialRequest,
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantRequest,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicLinearSystemRequest,
    SymbolicLinearSystemResult,
    SymbolicMatrix,
    SymbolicMatrixProductRequest,
    SymbolicMatrixRequest,
    SymbolicRankResult,
    _require_determinant_family_result_budget,
    _require_symbolic_product_admission,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction

SystemClassification = Literal["UNIQUE", "NON_UNIQUE", "INCONSISTENT"]

__all__ = [
    "SystemClassification",
    "symbolic_determinant",
    "symbolic_linear_system_solve",
    "symbolic_matrix_multiply",
    "symbolic_rank",
]


def _matrix_from_values(
    entries: tuple[tuple[RationalFunction, ...], ...],
) -> Any:
    import sympy

    if not entries or not entries[0]:
        raise ValueError("symbolic matrix must be nonempty")
    rows = len(entries)
    columns = len(entries[0])
    if any(len(row) != columns for row in entries):
        raise ValueError("symbolic matrix rows must all have the same length")
    if rows > 8 or columns > 8:
        raise ValueError("symbolic matrix dimensions must be between 1 and 8")
    return sympy.Matrix(
        [[rational_function_to_sympy(entry) for entry in row] for row in entries]
    )


def symbolic_determinant(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> RationalFunction:
    """Return the determinant in the declared rational-function field."""
    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("determinant requires a square matrix")
    return rational_function_from_sympy(matrix.det(method="bareiss"), variables)


def symbolic_rank(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> tuple[int, tuple[int, ...]]:
    """Return the exact symbolic rank and RREF pivot columns."""
    del variables
    matrix = _matrix_from_values(entries)
    _, pivots = matrix.rref()
    return len(pivots), tuple(int(c) for c in pivots)


def symbolic_matrix_multiply(
    left: SymbolicMatrix,
    right: SymbolicMatrix,
) -> SymbolicMatrix:
    """Return the exact product of compatible symbolic matrices.

    Native callers supply domain-owned ``SymbolicMatrix`` values. The owner
    admits complete work and result size before private SymPy multiplication.
    """

    _require_symbolic_product_admission(left, right)
    product = _matrix_from_values(left.entries) * _matrix_from_values(right.entries)
    return SymbolicMatrix(
        variables=left.variables,
        entries=tuple(
            tuple(
                rational_function_from_sympy(product[row, column], left.variables)
                for column in range(product.cols)
            )
            for row in range(product.rows)
        ),
    )


def symbolic_characteristic_polynomial(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> tuple[int, tuple[RationalFunction, ...]]:
    """Return (degree, descending coefficients) of det(lambda I - A)."""
    import sympy

    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    lam = sympy.Symbol("lambda")
    poly = (sympy.eye(matrix.rows) * lam - matrix).det(method="bareiss")
    expanded = sympy.Poly(poly, lam)
    coeffs = expanded.all_coeffs()
    return int(expanded.degree()), tuple(
        rational_function_from_sympy(coefficient, variables) for coefficient in coeffs
    )


def symbolic_eigenvalues(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> list[tuple[str, int]]:
    """Return a list of (eigenvalue_string, multiplicity) pairs."""
    from sympy import sstr

    del variables
    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("eigenvalues require a square matrix")
    eigenvalues = matrix.eigenvals()
    return [(sstr(value), int(mult)) for value, mult in eigenvalues.items()]


def _require_native_system_request(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
    variables: tuple[str, ...],
) -> None:
    """Validate the complete mathematical request for direct native callers.

    Applies the same shape, ordered-field consistency, and derived-solution
    budget checks as the wire request model before SymPy is invoked.
    """
    from jacobian.math.matrices.symbolic._models import (
        MAX_SYMBOLIC_MATRIX_DIMENSION,
        _require_linear_system_growth_admission,
    )

    if not entries:
        raise ValueError("symbolic matrix must be nonempty")
    # Wire matrix shape/dimension limits are applied BEFORE growth admission:
    # an oversized shape must be rejected up front instead of paying the
    # admission scan over a shape the wire envelope would never accept.
    rows = len(entries)
    columns = len(entries[0])
    if not columns:
        raise ValueError("symbolic matrix must be nonempty")
    if rows > MAX_SYMBOLIC_MATRIX_DIMENSION or columns > MAX_SYMBOLIC_MATRIX_DIMENSION:
        raise ValueError("symbolic matrix dimensions must be between 1 and 8")
    if any(len(row) != columns for row in entries):
        raise ValueError("symbolic matrix rows must all have the same length")
    if len(rhs) != len(entries):
        raise ValueError(
            "the right-hand side length must equal the coefficient row count"
        )
    for row in entries:
        for value in row:
            if value.variables != variables:
                raise ValueError("matrix entries must use the declared ordered field")
    for value in rhs:
        if value.variables != variables:
            raise ValueError("the right-hand side must use the declared ordered field")
    _require_linear_system_growth_admission(entries, rhs)


def symbolic_linear_system_solve(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
    variables: tuple[str, ...],
) -> tuple[
    SystemClassification,
    tuple[RationalFunction, ...] | None,
    tuple[RationalFunction, ...] | None,
    tuple[tuple[RationalFunction, ...], ...] | None,
]:
    """Solve ``A x = b`` over ``QQ(t_1, ..., t_n)``.

    Returns ``(classification, solution, particular_solution, nullspace_basis)``.
    """
    import sympy

    # Native callers bypass the wire envelope, so the complete mathematical
    # request is validated here before the backend runs.
    _require_native_system_request(entries, rhs, variables)

    matrix = _matrix_from_values(entries)
    rhs_vec = sympy.Matrix([[rational_function_to_sympy(v) for v in rhs]]).T

    rank_coeff = matrix.rank()
    aug = matrix.row_join(rhs_vec)
    rank_aug = aug.rank()
    n_cols = matrix.cols

    if rank_aug > rank_coeff:
        return "INCONSISTENT", None, None, None

    if rank_coeff == n_cols:
        # Unique solution. An exact backend failure here is an execution
        # failure, not a mathematical classification: conversion limits
        # prove nothing about consistency. LUsolve rejects overdetermined
        # rectangular systems, so read the unique solution from the exact
        # RREF of the augmented matrix, which handles any shape. With full
        # column rank and consistency already established, every coefficient
        # column is a pivot row and the augmented entry is the solution.
        rref_mat, pivots = aug.rref()
        solution: dict[int, RationalFunction] = {}
        for i, pivot_col in enumerate(pivots):
            if pivot_col < n_cols:
                solution[pivot_col] = rational_function_from_sympy(
                    rref_mat[i, n_cols], variables
                )
        if len(solution) != n_cols:
            raise ValueError(
                "exact row reduction did not determine the unique solution"
            )
        return "UNIQUE", tuple(solution[j] for j in range(n_cols)), None, None

    # Non-unique consistent system
    null = matrix.nullspace()
    nullspace_basis: tuple[tuple[RationalFunction, ...], ...] = tuple(
        tuple(rational_function_from_sympy(vec[i], variables) for i in range(n_cols))
        for vec in null
    )

    # Find a particular solution using the pseudo-inverse approach
    # or least squares.  For now, use sympy's linear solve.
    # Use the augmented matrix rref to find a particular solution
    rref_mat, pivots = aug.rref()
    # The particular solution: set free variables to 0
    # Extract from the RREF of the augmented matrix
    particular = []
    for _j in range(n_cols):
        particular.append(rational_function_from_sympy(sympy.Integer(0), variables))

    for i, pivot_col in enumerate(pivots):
        if pivot_col < n_cols:
            particular[pivot_col] = rational_function_from_sympy(
                rref_mat[i, n_cols], variables
            )

    return (
        "NON_UNIQUE",
        None,
        tuple(particular),
        nullspace_basis if nullspace_basis else None,
    )


def _domain_call[**P, R](call: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    try:
        return call(*args, **kwargs)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("request",), code=exc.type, message=exc.message()
        ) from exc
    except (ValueError, TypeError) as exc:
        raise OperationDomainValidationError(
            location=("request",), code="matrix.domain_invalid", message=str(exc)
        ) from exc


def _admit_determinant(request: SymbolicDeterminantRequest) -> None:
    _domain_call(
        _require_determinant_family_result_budget,
        request.matrix,
        characteristic_polynomial=False,
    )


def _admit_characteristic(request: SymbolicCharacteristicPolynomialRequest) -> None:
    _domain_call(
        _require_determinant_family_result_budget,
        request.matrix,
        characteristic_polynomial=True,
    )


def compute_symbolic_determinant(
    request: SymbolicDeterminantRequest,
) -> SymbolicDeterminantResult:
    _admit_determinant(request)
    determinant = symbolic_determinant(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicDeterminantResult(determinant=determinant)


def compute_symbolic_rank(
    request: SymbolicMatrixRequest,
) -> SymbolicRankResult:
    rank, pivot_columns = symbolic_rank(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicRankResult(rank=rank, pivot_columns=pivot_columns)


def compute_symbolic_matrix_product(
    request: SymbolicMatrixProductRequest,
) -> SymbolicMatrix:
    """Compute one exact symbolic matrix product."""

    return _domain_call(symbolic_matrix_multiply, request.left, request.right)


def compute_symbolic_characteristic_polynomial(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicCharacteristicPolynomialResult:
    _admit_characteristic(request)
    degree, coeffs = symbolic_characteristic_polynomial(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicCharacteristicPolynomialResult(
        degree=degree,
        coefficients_descending=tuple(coeffs),
    )


def compute_symbolic_eigenvalues(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicEigenvaluesResult:
    _admit_characteristic(request)
    entries = request.matrix.entries
    variables = request.matrix.variables
    try:
        eigenvalues = symbolic_eigenvalues(entries, variables)
    except MatrixError:
        # SymPy raises MatrixError when eigenvalues cannot be represented
        # in radicals.  Return the exact characteristic polynomial instead.
        degree, coeffs = symbolic_characteristic_polynomial(entries, variables)
        return SymbolicEigenvaluesResult(
            representation="ROOTS_BY_POLYNOMIAL",
            characteristic_polynomial=tuple(coeffs),
            degree=degree,
        )
    return SymbolicEigenvaluesResult(
        representation="EXPLICIT_ROOTS",
        eigenvalues=tuple(value for value, _ in eigenvalues),
        multiplicities=tuple(mult for _, mult in eigenvalues),
    )


def compute_symbolic_linear_system(
    request: SymbolicLinearSystemRequest,
) -> SymbolicLinearSystemResult:
    """Classify and solve ``A x = b`` over ``QQ(t_1, ..., t_n)``."""

    classification, solution, particular, nullspace = _domain_call(
        symbolic_linear_system_solve,
        request.matrix.entries,
        request.rhs,
        request.matrix.variables,
    )

    return SymbolicLinearSystemResult._from_kernel(
        system=request,
        classification=classification,
        solution=solution,
        particular_solution=particular,
        nullspace_basis=nullspace,
    )
