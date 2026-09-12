"""Exact canonical-form kernels backed by SymPy polynomial algebra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.canonical_forms._models import (
    MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    MAX_CANONICAL_FORM_DIMENSION,
    MAX_CANONICAL_FORM_SCALAR_DIGITS,
    MAX_MATRIX_POLYNOMIAL_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_REMAINDER_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS,
    InvariantFactorEntry,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
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
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "characteristic_polynomial",
    "evaluate_matrix_polynomial_value",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
    "reduce_matrix_polynomial",
    "verify_minimal_polynomial",
    "verify_primary_decomposition",
    "verify_rational_canonical_form",
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

    from sympy import Matrix, eye

    n = _square_dimension(entries)
    matrix = _sympy_matrix(entries)

    powers = [eye(n)]
    for _ in range(n):
        powers.append(powers[-1] * matrix)

    rows = [[mat[i, j] for i in range(n) for j in range(n)] for mat in powers]
    stacked = Matrix(rows).T

    reduced, pivots = stacked.rref()
    degree = next((index for index in range(n + 1) if index not in pivots), None)
    if degree is None:
        raise ArithmeticError("Krylov subspace exceeded the Cayley-Hamilton bound")
    if degree == 0:
        return (Fraction(1),)

    # All earlier power columns are pivots. The first free column gives the
    # unique monic dependence directly in the already-reduced Krylov matrix.
    return (
        *(Fraction(-reduced[index, degree]) for index in range(degree)),
        Fraction(1),
    )


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
    dimension = matrix.row_count
    if matrix.column_count != dimension or dimension == 0:
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
    rows = matrix.row_count
    columns = matrix.column_count
    if rows == 0:
        raise _validation_error(
            "budget_exceeded", "canonical-form operations require a nonempty matrix"
        )
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


def _to_monic_polynomial(
    coefficients: Sequence[Fraction], *, variable: str = "t"
) -> MonicPolynomial:
    from jacobian.math.polynomials.values import monic_polynomial_from_coefficients

    return monic_polynomial_from_coefficients(
        tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        ),
        variable=variable,
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


def _polynomial_from_coefficients(
    coefficients: Sequence[Fraction], variable: str
) -> RationalPolynomial:
    """Encode increasing-degree coefficients in the canonical sparse form."""

    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(coefficient),
            exponents=(degree,),
        )
        for degree, coefficient in reversed(tuple(enumerate(coefficients)))
        if coefficient
    )
    return RationalPolynomial(
        variables=(variable,), polynomial=SparseRationalPolynomial(terms=terms)
    )


def _divide_polynomials(
    dividend: RationalPolynomial, divisor: MonicPolynomial
) -> tuple[RationalPolynomial, RationalPolynomial]:
    """Perform exact univariate Euclidean division over QQ."""

    variable = dividend.variables[0]
    remainder = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in dividend.polynomial.terms
    }
    divisor_coefficients = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in divisor.polynomial.terms
    }
    divisor_degree = divisor.polynomial.terms[0].exponents[0]
    quotient: dict[int, Fraction] = {}
    while remainder and max(remainder) >= divisor_degree:
        degree = max(remainder)
        shift = degree - divisor_degree
        factor = remainder[degree]  # the divisor is monic
        quotient[shift] = quotient.get(shift, Fraction(0)) + factor
        for divisor_exponent, divisor_coefficient in divisor_coefficients.items():
            exponent = divisor_exponent + shift
            value = remainder.get(exponent, Fraction(0)) - factor * divisor_coefficient
            if value:
                remainder[exponent] = value
            else:
                remainder.pop(exponent, None)
    return (
        _polynomial_from_coefficients(
            [
                quotient.get(index, Fraction(0))
                for index in range(max(quotient, default=-1) + 1)
            ],
            variable,
        ),
        _polynomial_from_coefficients(
            [
                remainder.get(index, Fraction(0))
                for index in range(max(remainder, default=-1) + 1)
            ],
            variable,
        ),
    )


def reduce_matrix_polynomial(
    matrix: RationalMatrix, polynomial: RationalPolynomial
) -> tuple[MonicPolynomial, RationalPolynomial, RationalPolynomial]:
    """Return the minimal polynomial and exact quotient/remainder of ``polynomial``."""

    _admit_square(matrix)
    if len(polynomial.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="matrix.polynomial.remainder.variable",
            message="polynomial reduction requires exactly one variable",
        )
    try:
        require_polynomial_budget(
            polynomial,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="matrix polynomial remainder",
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="matrix.polynomial.remainder.budget",
            message=str(exc),
        ) from exc
    source_degree = max(
        (term.exponents[0] for term in polynomial.polynomial.terms), default=0
    )
    source_digits = max(
        (
            canonical_rational_component_digits(term.coefficient)
            for term in polynomial.polynomial.terms
        ),
        default=1,
    )
    # A zero or constant source needs no Euclidean quotient-growth work.  Keep
    # the square-matrix admission above and still compute the minimal
    # polynomial, which is part of this operation's result, before returning
    # the source polynomial unchanged as the remainder.
    if source_degree == 0:
        minimal_coefficients = minimal_polynomial(_matrix_entries(matrix))
        minimal = _to_monic_polynomial(
            minimal_coefficients, variable=polynomial.variables[0]
        )
        return (
            minimal,
            _polynomial_from_coefficients((), polynomial.variables[0]),
            polynomial,
        )
    # Euclidean division can fill every degree from zero through the leading
    # quotient degree. Reject a quotient whose canonical sparse carrier could
    # not hold that complete support before computing the matrix modulus.
    if source_degree > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="matrix.polynomial.remainder.output",
            message="polynomial remainder quotient support exceeds the canonical term bound",
        )
    # The characteristic polynomial bounds the minimal polynomial degree by n.
    # Clearing all n-entry products gives a coefficient-height bound of
    # n*(entry-height + 2) + ceil(log10(n+1)); the larger additive slack also
    # covers rational Krylov elimination used by the private producer.
    dimension = matrix.row_count
    matrix_digits = max(
        (
            canonical_rational_component_digits(entry)
            for row in matrix.entries
            for entry in row
        ),
        default=1,
    )
    minimal_digits_bound = dimension * (matrix_digits + 2) + dimension.bit_length() + 2
    # A recurrence step adds at most one minimal-polynomial coefficient times
    # a previous quotient coefficient. This is intentionally an upper bound:
    # it charges every source-degree step even when the source is sparse or
    # cancellation later reduces the exact result.
    estimated_digits = source_digits + (source_degree + 1) * (
        minimal_digits_bound + dimension.bit_length() + 2
    )
    digit_work = (
        len(polynomial.polynomial.terms) * (source_degree + 1) * (estimated_digits**2)
    )
    if estimated_digits > MAX_CANONICAL_RATIONAL_DIGITS or (
        digit_work > MAX_MATRIX_POLYNOMIAL_REMAINDER_DIGIT_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="matrix.polynomial.remainder.output",
            message="polynomial remainder quotient growth exceeds the admitted exact-arithmetic bound",
        )
    minimal_coefficients = minimal_polynomial(_matrix_entries(matrix))
    minimal = _to_monic_polynomial(
        minimal_coefficients, variable=polynomial.variables[0]
    )
    quotient, remainder = _divide_polynomials(polynomial, minimal)
    return minimal, quotient, remainder


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


def verify_minimal_polynomial(claim: MinimalPolynomialResult) -> bool:
    """Verify minimal and characteristic polynomials against the matrix."""

    if not isinstance(claim, MinimalPolynomialResult):
        return False
    try:
        minimal, characteristic = _minimal_polynomial_components(claim.matrix)
        expected = MinimalPolynomialResult._from_kernel(
            matrix=claim.matrix,
            minimal_polynomial=minimal,
            characteristic_polynomial=characteristic,
        )
        return expected == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_rational_canonical_form(claim: RationalCanonicalFormResult) -> bool:
    """Verify invariant factors and derived polynomials against the matrix."""

    if not isinstance(claim, RationalCanonicalFormResult):
        return False
    try:
        factors, characteristic, minimal = _rational_canonical_components(claim.matrix)
        expected = RationalCanonicalFormResult._from_kernel(
            matrix=claim.matrix,
            invariant_factors=factors,
            characteristic_polynomial=characteristic,
            minimal_polynomial=minimal,
        )
        return expected == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_primary_decomposition(claim: PrimaryDecompositionResult) -> bool:
    """Verify primary components and their product against the matrix."""

    if not isinstance(claim, PrimaryDecompositionResult):
        return False
    try:
        components, minimal = _primary_decomposition_components(claim.matrix)
        expected = PrimaryDecompositionResult._from_kernel(
            matrix=claim.matrix,
            components=components,
            minimal_polynomial=minimal,
        )
        return expected == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
