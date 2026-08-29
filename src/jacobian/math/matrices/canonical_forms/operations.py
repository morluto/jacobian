"""Exact canonical-form kernels backed by SymPy polynomial algebra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.canonical_forms._models import (
    MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    MAX_CANONICAL_FORM_DIMENSION,
    MAX_CANONICAL_FORM_SCALAR_DIGITS,
    MAX_MATRIX_POLYNOMIAL_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS,
    InvariantFactorEntry,
    MonicPolynomial,
    _polynomial_degree,
    _require_matrix_polynomial_output_budget,
    _validation_error,
)
from jacobian.math.matrices.values import (
    RationalMatrix,
    rational_matrix_from_fractions,
    require_matrix_scalar_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "characteristic_polynomial",
    "evaluate_matrix_polynomial_value",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
]

RationalEntries = Sequence[Sequence[Fraction]]
CoefficientList = tuple[Fraction, ...]


def _integer_decimal_digits(value: int) -> int:
    """Return the decimal width of ``value`` without decimal rendering.

    The conformance probe below can observe large admitted exact components.
    Avoiding ``str(int)`` keeps that private diagnostic independent of
    CPython's configurable decimal-conversion guard.
    """

    magnitude = abs(value)
    if magnitude == 0:
        return 1
    # 0.30103 is an upper rational approximation to log10(2), so this is an
    # upper estimate from the binary width.  A single exact comparison removes
    # the possible one-digit overestimate.
    digits = magnitude.bit_length() * 30_103 // 100_000 + 1
    if magnitude < 10 ** (digits - 1):
        return digits - 1
    return digits


@dataclass
class _HornerEvaluationMetrics:
    """Private kernel measurements for the matrix-polynomial pilot only.

    This is deliberately an optional observer on this owner-local kernel, not
    a cross-operation instrumentation protocol.  It records the published
    dense scalar-product proxy and the widest component of every materialized
    Horner state -- each pre-addition matrix product together with each
    reduced accumulator -- without changing evaluation semantics.  Inside a
    measured matrix product the observer additionally records every scalar
    multiplication term and every partial accumulation of the standard
    row-times-column dot products, so canceling terms are observed at their
    full product width before the following addition reduces them.
    """

    maximum_component_digits: int = 0
    scalar_product_terms: int = 0
    stored_states: int = 0

    def record_state(self, state: Any) -> None:
        """Record one materialized exact Horner state."""

        state_digits = max(
            (
                max(
                    _integer_decimal_digits(int(entry.p)),
                    _integer_decimal_digits(int(entry.q)),
                )
                for entry in state
            ),
            default=1,
        )
        self.maximum_component_digits = max(self.maximum_component_digits, state_digits)
        self.stored_states += 1


def _measured_matrix_product(
    result: Any,
    matrix: Any,
    dimension: int,
    metrics: _HornerEvaluationMetrics,
) -> Any:
    """Return ``result * matrix`` while observing inside each dot product.

    SymPy reduces every dot product fully before ``result * matrix``
    returns, so a plain product observer never sees canceling scalar-product
    terms: with ``A = H [[1, 1], [-1, -1]]`` the square ``A**2`` is zero even
    though each entry forms ``H**2`` and ``-H**2`` terms mid-multiplication.
    This helper walks the same standard row-times-column accumulation in one
    deterministic order (entries row-major, each inner index ascending), and
    SymPy still performs every entry multiplication and addition.  The
    returned matrix equals ``result * matrix`` exactly because reduced
    rational arithmetic is independent of the association order.
    """

    from sympy import Matrix, S

    rows: list[list[Any]] = []
    for row_index in range(dimension):
        row_entries: list[Any] = []
        for column_index in range(dimension):
            accumulator = S.Zero
            for inner_index in range(dimension):
                term = (
                    result[row_index, inner_index] * matrix[inner_index, column_index]
                )
                metrics.record_state((term,))
                accumulator = accumulator + term
                metrics.record_state((accumulator,))
            row_entries.append(accumulator)
        rows.append(row_entries)
    product = Matrix(rows)
    # The assembled product is materialized before the scalar term can
    # cancel it, so the observed bound must cover this state too.
    metrics.record_state(product)
    return product


def _square_dimension(entries: RationalEntries) -> int:
    """Return the shared side length of a nonempty square entry matrix."""

    dimension = len(entries)
    if dimension == 0:
        raise ValueError("canonical-form operations require a nonempty square matrix")
    if any(len(row) != dimension for row in entries):
        raise ValueError("canonical-form operations require a square matrix")
    return dimension


def _sympy_matrix(entries: RationalEntries) -> Any:
    from sympy import Matrix, Rational

    return Matrix(
        [
            [Rational(entry.numerator, entry.denominator) for entry in row]
            for row in entries
        ]
    )


def _to_fraction(value: Any) -> Fraction:
    from sympy import Rational

    if not isinstance(value, Rational):
        raise ValueError("canonical-form backend returned a non-rational value")
    return Fraction(int(value.p), int(value.q))


def _coefficients(poly: Any) -> CoefficientList:
    """Return a monic polynomial's increasing-degree rational coefficients."""

    return tuple(
        _to_fraction(coefficient) for coefficient in reversed(poly.all_coeffs())
    )


def characteristic_polynomial(entries: RationalEntries) -> CoefficientList:
    """Return the monic characteristic polynomial coefficients [a_0, ..., a_n]."""

    from sympy import Poly, Symbol

    x = Symbol("x")
    _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    charpoly = matrix.charpoly(x)
    return _coefficients(Poly(charpoly.as_expr(), x))


def _evaluate_polynomial(
    entries: RationalEntries,
    coefficients: Sequence[Fraction],
    *,
    metrics: _HornerEvaluationMetrics | None = None,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return ``f(A)`` for increasing-degree coefficients by exact Horner evaluation."""

    from sympy import Rational, eye, zeros

    dimension = _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    identity = eye(dimension)
    if not coefficients:
        result = zeros(dimension)
        if metrics is not None:
            metrics.record_state(result)
    else:
        leading = coefficients[-1]
        result = Rational(leading.numerator, leading.denominator) * identity
        if metrics is not None:
            metrics.record_state(result)
        for coefficient in reversed(coefficients[:-1]):
            scalar = Rational(coefficient.numerator, coefficient.denominator)
            if metrics is None:
                product = result * matrix
            else:
                # Each dot product reduces before ``result * matrix`` would
                # return, so measuring the plain product alone would miss
                # canceling scalar-product terms.  The measured expansion
                # records every term and partial accumulation of the same
                # standard row-times-column order and returns the identical
                # exact matrix.
                metrics.scalar_product_terms += dimension**3
                product = _measured_matrix_product(result, matrix, dimension, metrics)
            result = product + scalar * identity
            if metrics is not None:
                metrics.record_state(result)
    return tuple(
        tuple(_to_fraction(result[row, column]) for column in range(dimension))
        for row in range(dimension)
    )


def minimal_polynomial(entries: RationalEntries) -> CoefficientList:
    """Compute the minimal polynomial via the Krylov/nullspace method.

    Returns the monic minimal polynomial as coefficient list [a_0, ..., a_n].
    """

    from sympy import Matrix, Poly, Symbol, eye

    x = Symbol("x")
    n = _square_dimension(entries)
    matrix = _sympy_matrix(entries)

    powers = [eye(n)]
    for _ in range(n):
        powers.append(powers[-1] * matrix)

    rows = [[mat[i, j] for i in range(n) for j in range(n)] for mat in powers]
    stacked = Matrix(rows).T

    _reduced, pivots = stacked.rref()
    degree = next((index for index in range(n + 1) if index not in pivots), None)
    if degree is None:
        raise ArithmeticError("Krylov subspace exceeded the Cayley-Hamilton bound")
    if degree == 0:
        return (Fraction(1),)

    submatrix = stacked[:, : degree + 1]
    null_vectors = submatrix.nullspace()
    if not null_vectors:
        raise ArithmeticError(
            "Krylov subspace produced no minimal polynomial dependency"
        )

    coefficients = null_vectors[0]
    dependency = sum(coefficients[index] * x**index for index in range(degree + 1))
    return _coefficients(Poly(dependency, x).monic())


def invariant_factors(entries: RationalEntries) -> tuple[CoefficientList, ...]:
    """Compute the non-unit invariant factors over QQ[x].

    Returns a list of monic polynomial coefficient lists, ordered by divisibility:
    f_1 | f_2 | ... | f_s.
    """

    from sympy import QQ, Poly, Symbol, eye
    from sympy.matrices.normalforms import smith_normal_form

    x = Symbol("x")
    n = _square_dimension(entries)
    matrix = _sympy_matrix(entries)
    characteristic_matrix = x * eye(n) - matrix
    smith = smith_normal_form(characteristic_matrix, domain=QQ[x])

    factors: list[CoefficientList] = []
    for index in range(n):
        diagonal = smith[index, index]
        if diagonal == 0:
            continue
        factor = Poly(diagonal, x).monic()
        if factor.degree() >= 1:
            factors.append(_coefficients(factor))
    return tuple(factors)


def primary_decomposition(entries: RationalEntries) -> tuple[CoefficientList, ...]:
    """Decompose the minimal polynomial into irreducible-power components.

    Returns a list of monic polynomial coefficient lists, one for each
    irreducible factor raised to its multiplicity in the minimal polynomial.
    """

    from sympy import Poly, Symbol, factor_list

    x = Symbol("x")
    minimal_coefficients = minimal_polynomial(entries)
    minimal_expression = sum(
        coefficient * x**index for index, coefficient in enumerate(minimal_coefficients)
    )
    _constant, factors = factor_list(minimal_expression, x)

    components: list[CoefficientList] = []
    for factor, power in factors:
        monic = Poly(factor, x).monic()
        components.append(_coefficients(monic**power))
    return tuple(components)


def _admit_matrix_polynomial_evaluation(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> None:
    dimension = len(matrix.entries)
    if len(matrix.entries[0]) != dimension:
        raise _validation_error(
            "budget_exceeded",
            "matrix polynomial evaluation requires a square matrix",
        )
    if len(polynomial.variables) != 1:
        raise _validation_error(
            "budget_exceeded",
            "matrix polynomial evaluation requires exactly one polynomial variable",
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_POLYNOMIAL_TERMS,
        maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
        maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
        label="matrix polynomial",
    )
    degree = _polynomial_degree(polynomial)
    scalar_products_per_pass = degree * dimension**3
    total_scalar_products = (
        MATRIX_POLYNOMIAL_EVALUATION_PASSES * scalar_products_per_pass
    )
    if total_scalar_products > MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:
        raise _validation_error(
            "budget_exceeded",
            "matrix polynomial Horner evaluation and retained-source accounting "
            f"exceed the {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,}-scalar-product "
            "work bound",
        )
    maximum_arithmetic_digits = _require_matrix_polynomial_output_budget(
        matrix,
        polynomial,
        degree,
    )
    digit_work = total_scalar_products * maximum_arithmetic_digits**2
    if digit_work > MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "matrix polynomial exact-arithmetic work exceeds the coupled "
            f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}-unit digit-work bound",
        )


def _admit_square_matrix(matrix: RationalMatrix) -> None:
    rows = len(matrix.entries)
    columns = len(matrix.entries[0])
    if rows != columns:
        raise _validation_error(
            "budget_exceeded", "canonical-form operations require a square matrix"
        )
    if rows > MAX_CANONICAL_FORM_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            f"canonical-form operations are bounded to {MAX_CANONICAL_FORM_DIMENSION} x "
            f"{MAX_CANONICAL_FORM_DIMENSION} matrices",
        )
    require_matrix_scalar_digits(
        matrix.entries,
        maximum=MAX_CANONICAL_FORM_SCALAR_DIGITS,
        label="canonical-form matrix",
    )


def _admit_matrix_polynomial(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> None:
    try:
        _admit_matrix_polynomial_evaluation(matrix, polynomial)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("matrix",), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("matrix",), code="matrix.domain_invalid", message=str(exc)
        ) from exc


def _admit_square(matrix: RationalMatrix) -> None:
    try:
        _admit_square_matrix(matrix)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("matrix",), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("matrix",), code="matrix.domain_invalid", message=str(exc)
        ) from exc


def _matrix_entries(
    matrix: RationalMatrix,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(value.as_fraction() for value in row) for row in matrix.entries)


def _to_monic_polynomial(coefficients: Sequence[Fraction]) -> MonicPolynomial:
    return MonicPolynomial(
        coefficients=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        )
    )


def _dense_polynomial_coefficients(
    polynomial: RationalPolynomial,
) -> tuple[Fraction, ...]:
    degree = max(
        (term.exponents[0] for term in polynomial.polynomial.terms),
        default=0,
    )
    coefficients = [Fraction(0)] * (degree + 1)
    for term in polynomial.polynomial.terms:
        coefficients[term.exponents[0]] = term.coefficient.as_fraction()
    return tuple(coefficients)


def evaluate_matrix_polynomial_value(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> RationalMatrix:
    _admit_matrix_polynomial(matrix, polynomial)
    return _evaluate_matrix_polynomial_value(matrix, polynomial)


def _evaluate_matrix_polynomial_value(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> RationalMatrix:
    evaluated = _evaluate_polynomial(
        _matrix_entries(matrix),
        _dense_polynomial_coefficients(polynomial),
    )
    return rational_matrix_from_fractions(evaluated)


def _minimal_polynomial_components(
    matrix: RationalMatrix,
) -> tuple[MonicPolynomial, MonicPolynomial]:
    """Admit one matrix and compute its minimal and characteristic values."""

    _admit_square(matrix)
    entries = _matrix_entries(matrix)
    return (
        _to_monic_polynomial(minimal_polynomial(entries)),
        _to_monic_polynomial(characteristic_polynomial(entries)),
    )


def _rational_canonical_components(
    matrix: RationalMatrix,
) -> tuple[tuple[InvariantFactorEntry, ...], MonicPolynomial, MonicPolynomial]:
    """Admit one matrix and compute all rational-canonical components."""

    _admit_square(matrix)
    entries = _matrix_entries(matrix)
    factors = invariant_factors(entries)
    minimal = minimal_polynomial(entries)
    characteristic = characteristic_polynomial(entries)
    invariant_entries = tuple(
        InvariantFactorEntry(
            factor=_to_monic_polynomial(coefficients),
            block_size=len(coefficients) - 1,
        )
        for coefficients in factors
    )
    return (
        invariant_entries,
        _to_monic_polynomial(characteristic),
        _to_monic_polynomial(minimal),
    )


def _primary_decomposition_components(
    matrix: RationalMatrix,
) -> tuple[tuple[MonicPolynomial, ...], MonicPolynomial]:
    """Admit one matrix and compute its primary components and minimal value."""

    _admit_square(matrix)
    entries = _matrix_entries(matrix)
    components = primary_decomposition(entries)
    minimal_coefficients = [Fraction(1)]
    for component in components:
        product = [Fraction(0)] * (len(minimal_coefficients) + len(component) - 1)
        for left_index, left in enumerate(minimal_coefficients):
            for right_index, right in enumerate(component):
                product[left_index + right_index] += left * right
        minimal_coefficients = product
    return (
        tuple(_to_monic_polynomial(coefficient) for coefficient in components),
        _to_monic_polynomial(minimal_coefficients),
    )
