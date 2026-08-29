from __future__ import annotations

import random
from copy import deepcopy
from fractions import Fraction
from math import ceil, comb, log10, prod
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sympy import nextprime

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices._operation_models import (
    MAX_CHARACTERISTIC_POLYNOMIAL_ORDER,
    MAX_INPUT_SCALAR_DIGITS,
    CharacteristicPolynomialRequest,
)
from jacobian.math.matrices._tools import compute_characteristic_polynomial
from jacobian.math.matrices.canonical_forms._models import (
    _MAX_WORK_BOUND,
    _RESULT_ENTRY_OVERHEAD_BYTES,
    _RESULT_ENVELOPE_RESERVE_BYTES,
    MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    MAX_MATRIX_POLYNOMIAL_DIGIT_WORK,
    MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS,
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    _coefficient_ratios,
    _general_result_component_bounds,
    _linear_result_component_bounds,
    _polynomial_degree,
    _require_matrix_polynomial_output_budget,
    _work_exact_quotient,
)
from jacobian.math.matrices.canonical_forms._tools import (
    compute_matrix_polynomial_evaluation,
)
from jacobian.math.matrices.canonical_forms.operations import (
    _dense_polynomial_coefficients,
    _evaluate_polynomial,
    _HornerEvaluationMetrics,
)
from jacobian.math.matrices.operations import _admit_characteristic_polynomial
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

RationalInput = int | str | CanonicalRational | tuple[int | str, int | str]


def _rational(
    numerator: RationalInput, denominator: int | str = 1
) -> CanonicalRational:
    if isinstance(numerator, CanonicalRational):
        assert denominator == 1
        return numerator
    if isinstance(numerator, tuple):
        assert denominator == 1
        numerator, denominator = numerator
    return CanonicalRational(num=str(numerator), den=str(denominator))


def _matrix(*rows: tuple[RationalInput, ...]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(tuple(_rational(entry) for entry in row) for row in rows)
    )


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        limit = int(candidate**0.5)
        if all(candidate % prime for prime in primes if prime <= limit):
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def _pairwise_coprime_denominators(count: int, digits: int) -> list[int]:
    limit = 10**digits
    denominators: list[int] = []
    for prime in _first_primes(count):
        exponent = max(1, ceil((digits - 1) / log10(prime)))
        value = prime**exponent
        while value >= limit:
            exponent -= 1
            value = prime**exponent
        denominators.append(value)
    return denominators


def _distinct_primes(count: int, *, digits: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 10 ** (digits - 1)
    for _ in range(count):
        candidate = int(nextprime(candidate))
        primes.append(candidate)
    return tuple(primes)


def _diagonal_reciprocals(denominators: tuple[int, ...]) -> RationalMatrix:
    order = len(denominators)
    return _matrix(
        *tuple(
            tuple(
                (1, denominators[row]) if row == column else 0
                for column in range(order)
            )
            for row in range(order)
        )
    )


def _polynomial(*terms: tuple[RationalInput, int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=(exponent,),
                )
                for coefficient, exponent in terms
            )
        ),
    )


def _assert_admission_rejected(request: MatrixPolynomialEvaluationRequest) -> None:
    with pytest.raises(OperationDomainValidationError):
        compute_matrix_polynomial_evaluation(request.matrix, request.polynomial)


def _evaluate(
    request: MatrixPolynomialEvaluationRequest,
) -> MatrixPolynomialEvaluationResult:
    """Invoke the canonical native operation for a wire-shaped fixture."""

    return compute_matrix_polynomial_evaluation(request.matrix, request.polynomial)


def _fractions(matrix: RationalMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(entry.as_fraction() for entry in row) for row in matrix.entries)


def _matrix_polynomial_component_bounds(
    request: MatrixPolynomialEvaluationRequest,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Return the same owner-local result and intermediate estimates as admission."""

    degree = _polynomial_degree(request.polynomial)
    coefficients = _coefficient_ratios(request.polynomial)
    matrix_is_zero = all(
        entry.num == "0" for row in request.matrix.entries for entry in row
    )
    if degree <= 1 or matrix_is_zero:
        return _linear_result_component_bounds(request.matrix, coefficients)
    return _general_result_component_bounds(request.matrix, request.polynomial)


def _assert_matrix_polynomial_envelope_conformance(
    request: MatrixPolynomialEvaluationRequest,
) -> None:
    """Compare one public evaluation with the owner-local admission envelope."""

    component_bounds, _component_work_digits = _matrix_polynomial_component_bounds(
        request
    )
    estimated_work_digits = _require_matrix_polynomial_output_budget(
        request.matrix,
        request.polynomial,
        _polynomial_degree(request.polynomial),
    )

    metrics = _HornerEvaluationMetrics()
    measured_value = _evaluate_polynomial(
        _fractions(request.matrix),
        _dense_polynomial_coefficients(request.polynomial),
        metrics=metrics,
    )
    result = _evaluate(request)

    assert _fractions(result.value) == measured_value
    actual_component_bounds = tuple(
        (len(entry.num.lstrip("-")), len(entry.den))
        for row in result.value.entries
        for entry in row
    )
    assert all(
        actual_numerator_digits <= estimated_numerator_digits
        and actual_denominator_digits <= estimated_denominator_digits
        for (actual_numerator_digits, actual_denominator_digits), (
            estimated_numerator_digits,
            estimated_denominator_digits,
        ) in zip(actual_component_bounds, component_bounds, strict=True)
    )
    assert metrics.maximum_component_digits <= estimated_work_digits

    dimension = len(request.matrix.entries)
    degree = _polynomial_degree(request.polynomial)
    assert metrics.scalar_product_terms == degree * dimension**3
    # One initial state; per multiplication every scalar-product term and
    # partial accumulation inside each dot product is recorded (two states
    # per term) together with the assembled product and the reduced
    # accumulator.
    assert metrics.stored_states == 1 + degree * (2 * dimension**3 + 2)
    total_scalar_products = (
        MATRIX_POLYNOMIAL_EVALUATION_PASSES * metrics.scalar_product_terms
    )
    assert total_scalar_products <= MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS
    assert (
        total_scalar_products * estimated_work_digits**2
        <= MAX_MATRIX_POLYNOMIAL_DIGIT_WORK
    )

    estimated_value_bytes = sum(
        numerator_digits + denominator_digits + _RESULT_ENTRY_OVERHEAD_BYTES
        for numerator_digits, denominator_digits in component_bounds
    )
    estimated_output_bytes = (
        len(encode_strict_json(request.matrix.model_dump(mode="json")))
        + len(encode_strict_json(request.polynomial.model_dump(mode="json")))
        + estimated_value_bytes
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    actual_output_bytes = len(encode_strict_json(result.model_dump(mode="json")))
    assert actual_output_bytes <= estimated_output_bytes
    assert actual_output_bytes <= CanonicalLimits().max_output_bytes


@pytest.mark.parametrize(
    ("matrix", "polynomial"),
    [
        # Degenerate zero support keeps a high-degree Horner trace without a
        # matrix-side denominator product.
        (_matrix((0, 0), (0, 0)), _polynomial((1, 3), ((1, 3), 0))),
        # A chain clears dead powers after bounded shifted states.
        (
            _matrix((0, (1, 2), 0), (0, 0, (1, 3)), (0, 0, 0)),
            _polynomial(((1, 5), 5), ((1, 7), 4), (1, 1)),
        ),
        # Two routes meet at one cell, so coprime denominators compound in a
        # measured state rather than staying independent by construction.
        (
            _matrix(
                (0, (1, 2), (1, 3), 0),
                (0, 0, 0, (1, 5)),
                (0, 0, 0, (1, 7)),
                (0, 0, 0, 0),
            ),
            _polynomial((1, 2), ((1, 11), 1), ((1, 13), 0)),
        ),
        # Separate diagonal cells retain distinct rational components.
        (
            _matrix(((1, 2), 0), (0, (1, 3))),
            _polynomial(((1, 5), 4), ((1, 7), 2), ((1, 11), 0)),
        ),
        # Cyclic support exercises the conservative non-acyclic path.
        (
            _matrix((0, (1, 2)), ((1, 3), 0)),
            _polynomial((1, 5), ((1, 5), 3), ((1, 7), 0)),
        ),
    ],
    ids=("zero", "chain", "converging_paths", "disjoint_diagonal", "cycle"),
)
def test_matrix_polynomial_envelope_pilot_matches_measured_horner_states(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> None:
    """#2597 pilot: admitted representative shapes match their work/output envelope."""

    _assert_matrix_polynomial_envelope_conformance(
        MatrixPolynomialEvaluationRequest(matrix=matrix, polynomial=polynomial)
    )


def test_cancelled_horner_products_are_recorded_before_the_scalar_addition() -> None:
    # Scalar matrix A = [[H]] with f = t^2 - H*t: the first Horner
    # multiplication materializes H, yet adding c_1 = -H reduces the
    # accumulator to zero immediately, so every addition-reduced state stays
    # single-digit. The observer must charge the wide pre-addition product,
    # otherwise the conformance bound passes vacuously while admission
    # underestimates the shifted intermediate.
    height = 10**120
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((height,)),
        polynomial=_polynomial((1, 2), (-height, 1)),
    )

    metrics = _HornerEvaluationMetrics()
    measured_value = _evaluate_polynomial(
        _fractions(request.matrix),
        _dense_polynomial_coefficients(request.polynomial),
        metrics=metrics,
    )

    assert measured_value == ((Fraction(0),),)
    assert metrics.maximum_component_digits == len(str(height))
    assert metrics.stored_states == 1 + 2 * (2 * 1**3 + 2)

    estimated_work_digits = _require_matrix_polynomial_output_budget(
        request.matrix,
        request.polynomial,
        _polynomial_degree(request.polynomial),
    )
    assert len(str(height)) <= estimated_work_digits
    assert metrics.maximum_component_digits <= estimated_work_digits
    _assert_matrix_polynomial_envelope_conformance(request)


def test_canceling_dot_product_terms_are_observed_inside_each_product() -> None:
    # A = H [[1, 1], [-1, -1]] squares to zero: every entry of A^2 is a dot
    # product whose H^2 and -H^2 terms cancel, so recording only reduced
    # matrices observes nothing wider than H even though the multiplication
    # forms full H^2 terms mid-dot-product.  An admission regression that
    # underbounds scalar-product intermediates would then pass the
    # conformance assertion vacuously; the measured expansion must charge
    # each term and partial accumulation at its product width.
    height = 10**120
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((height, height), (-height, -height)),
        polynomial=_polynomial((1, 2)),
    )

    metrics = _HornerEvaluationMetrics()
    measured_value = _evaluate_polynomial(
        _fractions(request.matrix),
        _dense_polynomial_coefficients(request.polynomial),
        metrics=metrics,
    )

    assert measured_value == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert metrics.maximum_component_digits == len(str(height**2))
    assert metrics.stored_states == 1 + 2 * (2 * 2**3 + 2)

    estimated_work_digits = _require_matrix_polynomial_output_budget(
        request.matrix,
        request.polynomial,
        _polynomial_degree(request.polynomial),
    )
    assert len(str(height**2)) <= estimated_work_digits
    _assert_matrix_polynomial_envelope_conformance(request)


def _plain_sympy_horner_trace(
    entries: tuple[tuple[Fraction, ...], ...],
    coefficients: tuple[Fraction, ...],
) -> tuple[tuple[tuple[Fraction, ...], ...], int]:
    """Evaluate ``f(A)`` with plain SymPy matrix products, old-observer style.

    Returns the exact value together with the widest component among the
    pre-addition products and reduced accumulators -- exactly what recording
    whole matrices observed before dot products were expanded.
    """

    from sympy import Matrix, Rational, eye, zeros

    dimension = len(entries)
    matrix = Matrix(
        [
            [Rational(entry.numerator, entry.denominator) for entry in row]
            for row in entries
        ]
    )
    widest = 0

    def observe(state: Any) -> None:
        nonlocal widest
        widest = max(
            widest,
            max(
                max(len(str(abs(int(entry.p)))), len(str(int(entry.q))))
                for entry in state
            ),
        )

    if not coefficients:
        result = zeros(dimension)
        observe(result)
    else:
        result = Rational(
            coefficients[-1].numerator, coefficients[-1].denominator
        ) * eye(dimension)
        observe(result)
        for coefficient in reversed(coefficients[:-1]):
            scalar = Rational(coefficient.numerator, coefficient.denominator)
            observe(result * matrix)
            result = result * matrix + scalar * eye(dimension)
            observe(result)
    value = tuple(
        tuple(
            Fraction(int(result[row, column].p), int(result[row, column].q))
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    return value, widest


def test_measured_dot_product_expansion_matches_plain_sympy_multiplication() -> None:
    # Semantic neutrality of the observer expansion: on deterministic
    # pseudo-random shapes the instrumented evaluation returns exactly the
    # same rational matrix as plain ``result * matrix`` Horner evaluation,
    # while observing a superset of the plain whole-matrix states.
    rng = random.Random(20260826)
    cancellation_cases = (
        (
            ((Fraction(1), Fraction(1)), (Fraction(-1), Fraction(-1))),
            (Fraction(0), Fraction(0), Fraction(1)),
        ),
        (
            ((Fraction(1, 2), Fraction(1, 3)), (Fraction(-1, 3), Fraction(-1, 2))),
            (Fraction(0), Fraction(2), Fraction(1)),
        ),
    )
    random_cases = []
    for _case in range(6):
        dimension = rng.randint(1, 4)
        degree = rng.randint(0, 5)
        entries = tuple(
            tuple(
                Fraction(rng.randint(-6, 6), rng.choice((1, 1, 2, 3)))
                for _column in range(dimension)
            )
            for _row in range(dimension)
        )
        coefficients = tuple(
            Fraction(rng.randint(-9, 9), rng.choice((1, 2, 5)))
            for _exponent in range(degree + 1)
        )
        random_cases.append((entries, coefficients))

    for entries, coefficients in cancellation_cases + tuple(random_cases):
        plain_value, plain_widest = _plain_sympy_horner_trace(entries, coefficients)

        metrics = _HornerEvaluationMetrics()
        measured_value = _evaluate_polynomial(entries, coefficients, metrics=metrics)

        assert measured_value == plain_value
        assert metrics.maximum_component_digits >= plain_widest


@st.composite
def _small_horner_requests(
    draw: st.DrawFn,
) -> MatrixPolynomialEvaluationRequest:
    """Generate bounded support shapes around the representative pilot corpus."""

    dimension = draw(st.integers(min_value=1, max_value=3))
    support = draw(st.sampled_from(("zero", "chain", "diagonal", "dense")))
    numerator = st.integers(min_value=-3, max_value=3)
    denominator = st.sampled_from((1, 2, 3, 5))

    def entry(row: int, column: int) -> RationalInput:
        if support == "zero":
            return 0
        if support == "chain" and column != row + 1:
            return 0
        if support == "diagonal" and column != row:
            return 0
        value, divisor = draw(st.tuples(numerator, denominator))
        reduced = Fraction(value, divisor)
        return 0 if reduced == 0 else (reduced.numerator, reduced.denominator)

    matrix = _matrix(
        *(
            tuple(entry(row, column) for column in range(dimension))
            for row in range(dimension)
        )
    )
    degree = draw(st.integers(min_value=0, max_value=5))
    coefficients = [(draw(numerator), exponent) for exponent in range(degree, -1, -1)]
    return MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_polynomial(
            *(
                (coefficient, exponent)
                for coefficient, exponent in coefficients
                if coefficient
            )
        ),
    )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(_small_horner_requests())
def test_matrix_polynomial_envelope_pilot_covers_small_support_variation(
    request: MatrixPolynomialEvaluationRequest,
) -> None:
    """The pilot's conformance relation also holds across small support variation."""

    _assert_matrix_polynomial_envelope_conformance(request)


def test_saturated_work_estimates_do_not_reenter_exact_division() -> None:
    assert _work_exact_quotient(84, 7) == 12
    assert _work_exact_quotient(_MAX_WORK_BOUND + 1, 7) > _MAX_WORK_BOUND


def test_rotation_matrix_is_annihilated_by_t_squared_plus_one() -> None:
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, -1), (1, 0)),
        polynomial=_polynomial((1, 2), (1, 0)),
    )

    result = _evaluate(request)

    assert result.source_matrix == request.matrix
    assert result.polynomial == request.polynomial
    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 16


@pytest.mark.parametrize(
    ("polynomial", "expected"),
    [
        (_polynomial(), ((0, 0), (0, 0))),
        (_polynomial((3, 0)), ((3, 0), (0, 3))),
        (_polynomial((1, 1)), ((1, 2), (0, 1))),
    ],
)
def test_zero_constant_and_identity_polynomials(
    polynomial: RationalPolynomial,
    expected: tuple[tuple[int, ...], ...],
) -> None:
    result = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1, 2), (0, 1)),
            polynomial=polynomial,
        )
    )

    assert _fractions(result.value) == tuple(
        tuple(Fraction(entry) for entry in row) for row in expected
    )
    assert result.polynomial_degree == (
        None
        if not polynomial.polynomial.terms
        else polynomial.polynomial.terms[0].exponents[0]
    )


def test_evaluation_preserves_polynomial_sum_and_product() -> None:
    source = _matrix((1, 1), (0, 1))
    f = _polynomial((2, 1), (3, 0))
    g = _polynomial((1, 2), (-1, 0))
    polynomial_sum = _polynomial((1, 2), (2, 1), (2, 0))
    polynomial_product = _polynomial((2, 3), (3, 2), (-2, 1), (-3, 0))

    f_value = _fractions(
        _evaluate(MatrixPolynomialEvaluationRequest(matrix=source, polynomial=f)).value
    )
    g_value = _fractions(
        _evaluate(MatrixPolynomialEvaluationRequest(matrix=source, polynomial=g)).value
    )
    sum_value = _fractions(
        _evaluate(
            MatrixPolynomialEvaluationRequest(
                matrix=source,
                polynomial=polynomial_sum,
            )
        ).value
    )
    product_value = _fractions(
        _evaluate(
            MatrixPolynomialEvaluationRequest(
                matrix=source,
                polynomial=polynomial_product,
            )
        ).value
    )

    assert sum_value == tuple(
        tuple(f_value[row][column] + g_value[row][column] for column in range(2))
        for row in range(2)
    )
    assert product_value == tuple(
        tuple(
            sum(
                (f_value[row][inner] * g_value[inner][column] for inner in range(2)),
                start=Fraction(0),
            )
            for column in range(2)
        )
        for row in range(2)
    )


def test_value_composes_unchanged_with_matrix_consumers() -> None:
    evaluated = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((0, 1), (0, 0)),
            polynomial=_polynomial((1, 1), (1, 0)),
        )
    )

    characteristic = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=evaluated.value)
    )

    assert tuple(
        coefficient.as_fraction()
        for coefficient in characteristic.coefficients_descending
    ) == (Fraction(1), Fraction(-2), Fraction(1))


def test_flint_characteristic_polynomial_exceeds_shared_matrix_order() -> None:
    from math import factorial

    order = 96
    source = _matrix(
        *tuple(
            tuple(row + 1 if row == column else 0 for column in range(order))
            for row in range(order)
        )
    )

    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=source)
    )

    coefficients = tuple(
        coefficient.as_fraction() for coefficient in result.coefficients_descending
    )
    assert result.degree == order
    assert coefficients[0] == 1
    assert coefficients[1] == -sum(range(1, order + 1))
    assert coefficients[-1] == factorial(order)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            _matrix(((1, 2), 2), (0, (2, 3))),
            (Fraction(1), Fraction(-7, 6), Fraction(1, 3)),
        ),
        (
            _matrix((0, 1, 0), (0, 0, 1), (0, 0, 0)),
            (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        ),
    ],
)
def test_flint_characteristic_polynomial_preserves_exact_conventions(
    source: RationalMatrix, expected: tuple[Fraction, ...]
) -> None:
    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=source)
    )

    assert (
        tuple(
            coefficient.as_fraction() for coefficient in result.coefficients_descending
        )
        == expected
    )


def test_characteristic_polynomial_rejects_predicted_coefficient_growth() -> None:
    order = 128
    denominator = 10**255 + 1
    numerator = denominator - 1
    source = _matrix(
        *tuple(
            tuple(
                (numerator, denominator) if row == column else 0
                for column in range(order)
            )
            for row in range(order)
        )
    )

    with pytest.raises(OperationDomainValidationError, match="canonical digit budget"):
        compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(matrix=source)
        )


def test_characteristic_polynomial_rejects_coprime_denominator_lcm_growth() -> None:
    """Dense pairwise-coprime 256-digit dens exceed the budget after a few LCMs.

    Completing a 128x128 product LCM would materialize about four million
    digits; admission must stop from ``order * digits(LCM)`` before that.
    """
    order = MAX_CHARACTERISTIC_POLYNOMIAL_ORDER
    denominators = _pairwise_coprime_denominators(
        order * order, MAX_INPUT_SCALAR_DIGITS
    )
    source = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num="1", den=str(denominators[row * order + column]))
                for column in range(order)
            )
            for row in range(order)
        )
    )

    with pytest.raises(OperationDomainValidationError, match="canonical digit budget"):
        compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(matrix=source)
        )


def test_characteristic_polynomial_admits_shared_denominator_scaled_identity() -> None:
    order = 128
    source = _matrix(
        *tuple(
            tuple((1, 100) if row == column else 0 for column in range(order))
            for row in range(order)
        )
    )

    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=source)
    )

    coefficients = tuple(
        coefficient.as_fraction() for coefficient in result.coefficients_descending
    )
    expected = tuple(
        Fraction(comb(order, index) * ((-1) ** index), 100**index)
        for index in range(order + 1)
    )
    assert result.degree == order
    assert coefficients == expected


def test_characteristic_polynomial_admits_heterogeneous_100_digit_prime_diagonals() -> (
    None
):
    order = 32
    primes = _distinct_primes(order, digits=100)
    source = _diagonal_reciprocals(primes)

    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=source)
    )

    coefficients = tuple(
        coefficient.as_fraction() for coefficient in result.coefficients_descending
    )
    assert result.degree == order
    assert coefficients[0] == 1
    assert coefficients[1] == -sum(Fraction(1, prime) for prime in primes)
    assert coefficients[-1] == Fraction((-1) ** order, prod(primes))

    _admit_characteristic_polynomial(
        _diagonal_reciprocals(
            _distinct_primes(MAX_CHARACTERISTIC_POLYNOMIAL_ORDER, digits=100)
        )
    )


def test_characteristic_polynomial_admits_repeated_heterogeneous_rows() -> None:
    primes = _distinct_primes(32, digits=40)
    row = tuple((1, prime) for prime in primes)
    source = _matrix(*tuple(row for _ in range(32)))
    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(matrix=source)
    )
    total = sum(Fraction(1, prime) for prime in primes)
    coefficients = tuple(
        coefficient.as_fraction() for coefficient in result.coefficients_descending
    )
    assert coefficients[0] == 1
    assert coefficients[1] == -total
    assert all(coefficient == 0 for coefficient in coefficients[2:])


def test_characteristic_polynomial_stops_denominator_lcm_once_rejection_is_certain() -> (
    None
):
    order = MAX_CHARACTERISTIC_POLYNOMIAL_ORDER
    denominators = _pairwise_coprime_denominators(2 * order, digits=256)
    source = _matrix(
        *tuple(
            tuple(
                (1, denominators[row * order + column]) if row < 2 else 0
                for column in range(order)
            )
            for row in range(order)
        )
    )

    with pytest.raises(OperationDomainValidationError, match="canonical digit budget"):
        compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(matrix=source)
        )


def test_native_characteristic_polynomial_shares_widened_flint_kernel() -> None:
    import sympy

    from jacobian.math import matrices

    order = 96
    source = sympy.diag(*range(1, order + 1))

    polynomial = matrices.characteristic_polynomial(source, "lambda")
    wire = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(
            matrix=RationalMatrix(
                entries=tuple(
                    tuple(
                        CanonicalRational.from_integer_ratio(
                            int(source[row, column]), 1
                        )
                        for column in range(order)
                    )
                    for row in range(order)
                )
            )
        )
    )

    assert polynomial.all_coeffs() == [
        coefficient.as_fraction() for coefficient in wire.coefficients_descending
    ]


def test_adapter_preserves_canonical_coefficients_above_python_digit_limit() -> None:
    numerator = "1" * 5_000
    result = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=_polynomial((numerator, 0)),
        )
    )

    assert result.value.entries[0][0].num == numerator
    assert result.value.entries[0][0].den == "1"


@pytest.mark.parametrize(
    "mutation",
    ["value", "matrix", "polynomial", "degree", "matrix_work", "scalar_work"],
)
def test_result_rejects_independent_source_value_and_work_mutations(
    mutation: str,
) -> None:
    result = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((0, -1), (1, 0)),
            polynomial=_polynomial((1, 2), (1, 0)),
        )
    )
    wire = deepcopy(result.model_dump(mode="json"))
    if mutation == "value":
        wire["value"]["entries"][0][0] = {"num": "1", "den": "1"}
    elif mutation == "matrix":
        wire["source_matrix"]["entries"][0][0] = {"num": "1", "den": "1"}
    elif mutation == "polynomial":
        wire["polynomial"]["polynomial"]["terms"][1]["coefficient"] = {
            "num": "2",
            "den": "1",
        }
    elif mutation == "degree":
        wire["polynomial_degree"] = 1
    elif mutation == "matrix_work":
        wire["matrix_multiplications"] = 1
    else:
        wire["scalar_product_terms"] = 1

    if mutation in {"degree", "matrix_work", "scalar_work"}:
        with pytest.raises(ValidationError):
            MatrixPolynomialEvaluationResult.model_validate(wire)
    else:
        MatrixPolynomialEvaluationResult.model_validate(wire)


def test_request_rejects_non_square_and_multivariate_sources() -> None:
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1, 2)),
            polynomial=_polynomial((1, 1)),
        )
    )

    multivariate = RationalPolynomial(
        variables=("s", "t"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=_rational(1),
                    exponents=(1, 0),
                ),
            )
        ),
    )
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=multivariate,
        )
    )


def test_horner_work_boundary_is_derived_from_degree_and_matrix_order() -> None:
    zero = _rational(0)
    one = _rational(1)
    identity = RationalMatrix(
        entries=tuple(
            tuple(one if row == column else zero for column in range(32))
            for row in range(32)
        )
    )

    maximum_degree = (
        MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS
        // MATRIX_POLYNOMIAL_EVALUATION_PASSES
        // 32**3
    )
    accepted = MatrixPolynomialEvaluationRequest(
        matrix=identity,
        polynomial=_polynomial((1, maximum_degree)),
    )
    assert accepted.polynomial.polynomial.terms[0].exponents == (maximum_degree,)

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=identity,
            polynomial=_polynomial((1, maximum_degree + 1)),
        )
    )


def test_work_admission_couples_products_to_exact_component_growth() -> None:
    moderate = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((10**30,)),
        polynomial=_polynomial((1, 500)),
    )
    assert moderate.polynomial.polynomial.terms[0].exponents == (500,)

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((10**30,)),
            polynomial=_polynomial((1, 1_000)),
        )
    )


def test_result_sensitive_admission_accepts_maximum_sparse_exponent_at_one_by_one() -> (
    None
):
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((1,)),
        polynomial=_polynomial((1, 32_768)),
    )
    assert request.polynomial.polynomial.terms[0].exponents == (32_768,)


def test_zero_matrix_admission_does_not_combine_irrelevant_denominators() -> None:
    first_denominator = "1" + "0" * 20_000
    second_denominator = format_canonical_integer(3**42_000)

    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0,)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(2,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(1,),
                    ),
                )
            ),
        ),
    )

    assert request.matrix.entries[0][0].num == "0"


def test_request_rejects_predicted_scalar_and_aggregate_output_overflow() -> None:
    denominator = "1" + "0" * 100
    overflowing_exponent = MAX_CANONICAL_RATIONAL_DIGITS // 100 + 1
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(1, denominator),),)),
            polynomial=_polynomial((1, overflowing_exponent)),
        )
    )

    overflowing_entry_digits = CanonicalLimits().max_output_bytes // 32**2 + 1_000
    huge_coefficient = "1" * overflowing_entry_digits
    dense = RationalMatrix(
        entries=tuple(tuple(_rational(1) for _ in range(32)) for _ in range(32))
    )
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=dense,
            polynomial=_polynomial((huge_coefficient, 1)),
        )
    )


def test_structurally_nilpotent_powers_are_admitted_and_evaluate_to_zero() -> None:
    height = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(height)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2)),
    )

    result = _evaluate(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 16


def test_height_maximum_is_not_cancelled_from_the_result_bound() -> None:
    # diag(1, 1/q) with f = t^2 clears to Q = h = q, but h is only the
    # maximum cleared height, not a factor of every M^k entry: the true
    # second diagonal entry is 1/q^2 with compounded denominator digits.
    # The squared output must be predicted and rejected during request
    # validation instead of passing admission and failing result conversion.
    denominator = "1" + "0" * 17_000
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(1), _rational(0)),
                    (_rational(0), _rational(1, denominator)),
                )
            ),
            polynomial=_polynomial((1, 2)),
        )
    )


def test_dead_powers_do_not_demand_a_global_clearing_denominator() -> None:
    # A square-zero matrix whose only entries carry coprime 17,000-digit
    # denominators: t^2 is structurally zero, so no global LCM of entry
    # denominators may be required before the support analysis runs.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(7**20_118)
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (
                    _rational(0),
                    _rational(1, first_denominator),
                    _rational(1, second_denominator),
                ),
                (_rational(0), _rational(0), _rational(0)),
                (_rational(0), _rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2)),
    )

    result = _evaluate(request)

    assert _fractions(result.value) == tuple(
        tuple(Fraction(0) for _ in range(3)) for _ in range(3)
    )
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2
    assert result.scalar_product_terms == 54


def test_proven_cancellations_survive_with_compounded_denominators() -> None:
    # Swapped denominators put the full compounded height into one cleared
    # row: h = q^2. With a matching lifted coefficient the proven factor
    # cancels q^2 exactly, so both requests stay admitted while the bounds
    # still charge the compounded denominators honestly.
    base = format_canonical_integer(2**12_000)
    swap = RationalMatrix(
        entries=(
            (_rational(0), _rational(base)),
            (_rational(1, base), _rational(0)),
        )
    )

    identity_result = _evaluate(
        MatrixPolynomialEvaluationRequest(matrix=swap, polynomial=_polynomial((1, 2)))
    )
    assert _fractions(identity_result.value) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )

    scaled_result = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=swap,
            polynomial=_polynomial((format_canonical_integer(2**24_000), 2)),
        )
    )
    assert scaled_result.value.entries[0][0].num == format_canonical_integer(2**24_000)
    assert scaled_result.value.entries[1][1].num == format_canonical_integer(2**24_000)
    assert scaled_result.value.entries[0][1].num == "0"


def test_unprovable_height_growth_is_rejected_during_request_validation() -> None:
    # Same swapped shape with a larger base and no cancellable coefficient:
    # the honest compounded prediction n * h^2 exceeds the canonical component
    # cap even though the exact value would be the identity. Admission cannot
    # establish the tighter claim, so the request must be rejected here rather
    # than admitted and rescued by result conversion.
    base = format_canonical_integer(2**40_000)
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(base)),
                    (_rational(1, base), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 2)),
        )
    )


def test_structural_zero_reduces_to_surviving_terms_exactly() -> None:
    height = format_canonical_integer(7**18_000)
    nilpotent = RationalMatrix(
        entries=(
            (_rational(0), _rational(height)),
            (_rational(0), _rational(0)),
        )
    )
    with_constant = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=nilpotent,
            polynomial=_polynomial((1, 2), (5, 0)),
        )
    )
    assert _fractions(with_constant.value) == (
        (Fraction(5), Fraction(0)),
        (Fraction(0), Fraction(5)),
    )

    fractional = RationalMatrix(
        entries=((_rational(0), _rational(1, 3)), (_rational(0), _rational(0)))
    )
    rational_constant = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=fractional,
            polynomial=_polynomial((1, 2), (5, 0)),
        )
    )
    assert _fractions(rational_constant.value) == (
        (Fraction(5), Fraction(0)),
        (Fraction(0), Fraction(5)),
    )

    surviving_power = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=nilpotent,
            polynomial=_polynomial((1, 2), (1, 1)),
        )
    )
    assert surviving_power.value.entries[0][1].num == height
    assert surviving_power.value.entries[1][0].num == "0"


def test_admission_still_charges_live_structural_growth() -> None:
    height = "1" + "0" * 20_000
    cyclic = RationalMatrix(
        entries=(
            (_rational(0), _rational(height)),
            (_rational(height), _rational(0)),
        )
    )
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(matrix=cyclic, polynomial=_polynomial((1, 2)))
    )

    chain = RationalMatrix(
        entries=(
            (_rational(0), _rational(height), _rational(0)),
            (_rational(0), _rational(0), _rational(height)),
            (_rational(0), _rational(0), _rational(0)),
        )
    )
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 2)))
    )

    vanishing_chain_value = _evaluate(
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 3)))
    )
    assert _fractions(vanishing_chain_value.value) == (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
    )


def test_request_schema_publishes_coupled_digit_work_bound() -> None:
    schema = MatrixPolynomialEvaluationRequest.model_json_schema()

    polynomial_description = schema["properties"]["polynomial"]["description"]
    assert "(2 * degree * order^3) scalar products" in polynomial_description
    assert "largest decimal-digit component" in polynomial_description
    assert "predicted shifted Horner intermediate components" in polynomial_description
    assert f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}" in polynomial_description


def _rational_polynomial(
    *terms: tuple[RationalInput, int],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=(exponent,),
                )
                for coefficient, exponent in terms
            )
        ),
    )


def test_degree_two_admission_cross_cancels_coefficient_and_matrix_power_factors() -> (
    None
):
    base_two = format_canonical_integer(2**53_179)
    base_three = format_canonical_integer(3**33_558)
    coefficient = _rational(
        format_canonical_integer(2**106_358),
        format_canonical_integer(3**67_116),
    )

    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(base_three, base_two),),)),
        polynomial=_rational_polynomial((coefficient, 2)),
    )
    result = _evaluate(request)

    assert request.polynomial.polynomial.terms[0].exponents == (2,)
    assert result.value.entries[0][0].num == "1"
    assert result.value.entries[0][0].den == "1"
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2


def test_degree_two_admission_still_rejects_uncancellable_power_growth() -> None:
    entry_numerator = format_canonical_integer(7**24_048)
    entry_denominator = format_canonical_integer(5**28_072)
    coefficient_numerator = format_canonical_integer(11**18_900)
    coefficient_denominator = format_canonical_integer(13**17_400)

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=((_rational(entry_numerator, entry_denominator),),)
            ),
            polynomial=_rational_polynomial(
                (
                    _rational(coefficient_numerator, coefficient_denominator),
                    2,
                )
            ),
        )
    )


def test_admission_falls_back_to_dense_bound_beyond_materialization_ceiling() -> None:
    height = "1" + "0" * 20_000
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(height),),)),
            polynomial=_polynomial((1, 5)),
        )
    )

    moderate = "1" + "0" * 15_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(moderate),),)),
        polynomial=_polynomial((1, 2)),
    )
    assert request.matrix.entries[0][0].num == moderate


def test_constant_result_work_estimates_are_not_clipped_at_the_component_cap() -> None:
    # A 4x4 superdiagonal chain of 32,768-digit entries with f = t^4: every
    # nonconstant power is structurally dead, yet Horner still materializes
    # an A^3 entry compounding three input heights (~98,304 digits) on the
    # way to the exact zero value. Clipping the shifted-height work proxy at
    # one canonical component would predict 32,769 digits and admit about
    # 5.5e11 digit-work units while execution performs about 4.9e12, so the
    # unclipped work estimate must reject this request during validation
    # while the same shape at heights whose compounded shifts fit the
    # coupled budget stays admitted.
    height = "1" + "0" * 32_767
    chain = RationalMatrix(
        entries=tuple(
            tuple(
                _rational(height) if column == row + 1 else _rational(0)
                for column in range(4)
            )
            for row in range(4)
        )
    )

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(matrix=chain, polynomial=_polynomial((1, 4)))
    )

    moderate_height = "1" + "0" * 13_999
    admitted = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (
                        _rational(0),
                        _rational(moderate_height),
                        _rational(0),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(moderate_height),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(moderate_height),
                    ),
                    (_rational(0), _rational(0), _rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 4)),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(4)) for _ in range(4)
    )
    assert admitted.matrix_multiplications == 4


def test_structurally_dead_powers_are_excluded_from_digit_work_estimate() -> None:
    height = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(height)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 100)),
    )

    result = _evaluate(request)

    assert request.polynomial.polynomial.terms[0].exponents == (100,)
    assert result.polynomial_degree == 100
    assert result.matrix_multiplications == 100
    assert result.scalar_product_terms == 800
    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )


def test_horner_charges_shifted_dead_leading_terms_during_their_ride() -> None:
    # The live t term keeps the square-zero support in the general branch,
    # and Horner materializes the doubled-height entry C*H on its first
    # multiplication before later shifts clear the structurally dead C*t^50.
    # Dropping the dead leading term from work accounting would admit this
    # request on a 20,001-digit prediction while execution constructs a
    # 40,000-digit product, so honest charging must reject it here while the
    # same shape at smaller heights stays admitted and evaluates exactly.
    height = "1" + "0" * 20_000
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(height)),
                    (_rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial(("1" + "0" * 20_000, 50), (1, 1)),
        )
    )

    moderate_height = "1" + "0" * 14_999
    admitted = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (_rational(0), _rational(moderate_height)),
                    (_rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial(("1" + "0" * 14_999, 50), (1, 1)),
        )
    )

    assert admitted.value.entries[0][1].num == moderate_height
    assert admitted.value.entries[0][0].num == "0"
    assert admitted.polynomial_degree == 50
    assert admitted.matrix_multiplications == 50


def test_clearing_denominator_growth_is_bounded_by_live_horner_shifts() -> None:
    # Square-zero [[0, 1/q], [0, 0]] with a 20,001-digit q and f = t^100 + t:
    # the dead leading power clears on the first multiplication, so every
    # Horner intermediate stays 0, I, or A and no denominator ever exceeds
    # q. Raising the clearing denominator to the raw ordinary degree would
    # predict q^100 (~two million work digits) and reject a request whose
    # documented proxy 1600 * 20001^2 stays below the coupled budget;
    # bounding denominator growth by the maximum live Horner shift admits
    # it, and f(A) equals A exactly.
    denominator = "1" + "0" * 20_000
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (_rational(0), _rational(1, denominator)),
                (_rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 100), (1, 1)),
    )

    result = _evaluate(request)

    assert result.value.entries[0][0].num == "0"
    assert result.value.entries[1][0].num == "0"
    assert result.value.entries[1][1].num == "0"
    assert result.value.entries[0][1].num == "1"
    assert result.value.entries[0][1].den == denominator
    assert result.polynomial_degree == 100
    assert result.matrix_multiplications == 100
    assert result.scalar_product_terms == 800


def test_overlapping_dead_denominators_are_charged_at_shared_entries() -> None:
    # Acyclic chain A = [[0,1,1],[0,0,1],[0,0,0]] with f = t^5/a + t^4/b for
    # coprime 32,768-digit denominators: both powers are structurally dead,
    # yet Horner temporarily forms (A^2)[0,2]/a + A[0,2]/b at the shared
    # entry [0,2], whose reduced denominator compounds to about 65,535
    # digits. Predicting only the largest single denominator admits about
    # 2.9e11 digit-work units while execution performs about 1.16e12, so the
    # resolved per-entry charge must reject this request while the same
    # shape at denominators whose compounded width fits the coupled budget
    # stays admitted and evaluates to the exact zero matrix.
    first_denominator = "1" + "0" * 32_767
    second_denominator = format_canonical_integer(11**31_400)
    chain = _matrix((0, 1, 1), (0, 0, 1), (0, 0, 0))

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, first_denominator), 5),
                (_rational(1, second_denominator), 4),
            ),
        )
    )

    moderate_first = "1" + "0" * 16_383
    moderate_second = format_canonical_integer(11**15_700)
    admitted = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, moderate_first), 5),
                (_rational(1, moderate_second), 4),
            ),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(3)) for _ in range(3)
    )
    assert admitted.polynomial_degree == 5


def test_disjoint_dead_denominators_never_compound_across_entries() -> None:
    # Square-zero support with a live t term and f = t^3/a + t^2/b + t for
    # coprime 17,000-digit a and b: during the ride the 1/a term occupies
    # the off-diagonal while 1/b occupies the diagonal, and the former dies
    # before the latter shifts off-diagonal, so no entry ever combines the
    # denominators and every intermediate stays within 17,000 digits. A
    # global dead-term lcm would exceed the canonical cap and reject this
    # safely bounded request; resolved coexistence must admit it with
    # f(A) = A exactly.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, 1), (0, 0)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(3,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(2,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1),
                        exponents=(1,),
                    ),
                )
            ),
        ),
    )

    result = _evaluate(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 3
    assert result.matrix_multiplications == 3


def test_mixed_overlap_of_dead_denominators_is_still_charged() -> None:
    # The disjointness relief above must not drop genuine overlap charging:
    # the same chain shape with a surviving t term rides t^5/a and t^4/b
    # through the shared entry [0,2] exactly as in the constant case, so
    # coprime 31,000-digit denominators compound past the digit-work budget
    # and must be rejected, while the smaller honest twin stays admitted
    # with its exact value preserved.
    first_denominator = "1" + "0" * 31_000
    second_denominator = format_canonical_integer(11**29_800)
    chain = _matrix((0, 1, 1), (0, 0, 1), (0, 0, 0))

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, first_denominator), 5),
                (_rational(1, second_denominator), 4),
                (_rational(1), 1),
            ),
        )
    )

    moderate_first = "1" + "0" * 8_000
    moderate_second = format_canonical_integer(11**7_700)
    admitted = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=chain,
            polynomial=_rational_polynomial(
                (_rational(1, moderate_first), 5),
                (_rational(1, moderate_second), 4),
                (_rational(1), 1),
            ),
        )
    )

    assert _fractions(admitted.value) == (
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(0)),
    )


def test_converging_matrix_paths_compound_shared_cell_denominators() -> None:
    # A 4x4 diamond whose edges carry reciprocals of four pairwise-coprime
    # 25,000-digit integers with f = t^3: every power is structurally dead
    # and the exact value is zero, yet Horner materializes
    # (A^2)[0,3] = 1/(q1*q2) + 1/(q3*q4) at the shared cell [0,3], whose
    # reduced denominator compounds to about 100,000 digits. Predicting
    # entry_height^2 (about 50,000 digits) passes the coupled digit-work
    # check while the true intermediate exceeds it, so the walk-denominator
    # resolution must reject this request; the same shape at smaller
    # denominators stays admitted and evaluates exactly to zero.
    first = "1" + "0" * 24_999
    second = format_canonical_integer(3**52_408)
    third = format_canonical_integer(7**29_586)
    fourth = format_canonical_integer(11**24_007)
    diamond = RationalMatrix(
        entries=(
            (_rational(0), _rational(1, first), _rational(1, second), _rational(0)),
            (_rational(0), _rational(0), _rational(0), _rational(1, third)),
            (_rational(0), _rational(0), _rational(0), _rational(1, fourth)),
            (_rational(0), _rational(0), _rational(0), _rational(0)),
        )
    )

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=diamond,
            polynomial=_polynomial((1, 3)),
        )
    )

    small_first = format_canonical_integer(3**12_578)
    small_second = format_canonical_integer(7**7_117)
    small_third = format_canonical_integer(11**5_834)
    small_fourth = format_canonical_integer(13**5_332)
    admitted = _evaluate(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=(
                    (
                        _rational(0),
                        _rational(1, small_first),
                        _rational(1, small_second),
                        _rational(0),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(1, small_third),
                    ),
                    (
                        _rational(0),
                        _rational(0),
                        _rational(0),
                        _rational(1, small_fourth),
                    ),
                    (_rational(0), _rational(0), _rational(0), _rational(0)),
                )
            ),
            polynomial=_polynomial((1, 3)),
        )
    )

    assert _fractions(admitted.value) == tuple(
        tuple(Fraction(0) for _ in range(4)) for _ in range(4)
    )
    assert admitted.polynomial_degree == 3


def test_disjoint_rational_entries_never_demand_a_global_clearing_denominator() -> None:
    # Square-zero support with two rational entries in one row and a live t
    # term: f = t^2 + t with coprime 17,000-digit entry denominators keeps
    # the two denominators in separate entries at every Horner state and in
    # the exact value itself, so no per-cell coexistence ever compounds them.
    # Forming the global clearing lcm(a, b) exceeds the canonical cap and
    # would reject this request although every input, intermediate, output,
    # and digit-work bound holds; resolved coexistence admits it with
    # f(A) = A exactly.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(
            entries=(
                (
                    _rational(0),
                    _rational(1, first_denominator),
                    _rational(1, second_denominator),
                ),
                (_rational(0), _rational(0), _rational(0)),
                (_rational(0), _rational(0), _rational(0)),
            )
        ),
        polynomial=_polynomial((1, 2), (1, 1)),
    )

    result = _evaluate(request)

    assert result.value.entries[0][1].num == "1"
    assert result.value.entries[0][1].den == first_denominator
    assert result.value.entries[0][2].den == second_denominator
    assert result.value.entries[1][0].num == "0"
    assert result.polynomial_degree == 2
    assert result.matrix_multiplications == 2


def test_dead_coefficient_powers_are_classified_before_the_coefficient_lcm() -> None:
    # Square-zero [[0, 1], [0, 0]] with f = t^3/a + t^2/b for coprime
    # 17,000-digit a and b: both nonconstant powers are structurally dead
    # and the exact value is zero, so forming lcm(a, b) before support
    # classification would reject a request whose every Horner intermediate
    # stays within a single input denominator.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)
    request = MatrixPolynomialEvaluationRequest(
        matrix=_matrix((0, 1), (0, 0)),
        polynomial=RationalPolynomial(
            variables=("t",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=_rational(1, first_denominator),
                        exponents=(3,),
                    ),
                    RationalPolynomialTerm(
                        coefficient=_rational(1, second_denominator),
                        exponents=(2,),
                    ),
                )
            ),
        ),
    )

    result = _evaluate(request)

    assert _fractions(result.value) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert result.polynomial_degree == 3
    assert result.matrix_multiplications == 3
    assert result.scalar_product_terms == 24


def test_surviving_coefficient_denominators_still_demand_a_common_denominator() -> None:
    # The same coprime denominators attached to powers that reach the value
    # compound genuinely: the linear identity case produces (a + b)/(ab) on
    # the diagonal, and the general degree-2 case clears its surviving
    # coefficients through lcm(a, b). Both compounded predictions exceed the
    # canonical cap, so admission must still reject during validation.
    first_denominator = "1" + "0" * 17_000
    second_denominator = format_canonical_integer(11**16_325)

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((1,)),
            polynomial=RationalPolynomial(
                variables=("t",),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=_rational(1, first_denominator),
                            exponents=(1,),
                        ),
                        RationalPolynomialTerm(
                            coefficient=_rational(1, second_denominator),
                            exponents=(0,),
                        ),
                    )
                ),
            ),
        )
    )

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=_matrix((2, 0), (0, 2)),
            polynomial=RationalPolynomial(
                variables=("t",),
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=_rational(1, first_denominator),
                            exponents=(2,),
                        ),
                        RationalPolynomialTerm(
                            coefficient=_rational(1, second_denominator),
                            exponents=(1,),
                        ),
                    )
                ),
            ),
        )
    )


def test_linear_admission_cross_cancels_rational_product_factors() -> None:
    numerator = format_canonical_integer(2**66_037)
    denominator = format_canonical_integer(3**42_017)
    matrix = RationalMatrix(entries=((_rational(denominator, numerator),),))
    coefficient = _rational(numerator, denominator)

    request = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_rational_polynomial((coefficient, 1)),
    )
    result = _evaluate(request)

    assert request.polynomial.polynomial.terms[0].exponents == (1,)
    assert result.value.entries[0][0].num == "1"
    assert result.value.entries[0][0].den == "1"

    cancelled_with_constant = MatrixPolynomialEvaluationRequest(
        matrix=matrix,
        polynomial=_rational_polynomial((coefficient, 1), (_rational(5, 7), 0)),
    )
    constant_result = _evaluate(cancelled_with_constant)

    assert constant_result.value.entries[0][0].as_fraction() == Fraction(12, 7)


def test_linear_admission_still_rejects_uncancellable_product_growth() -> None:
    coefficient_numerator = format_canonical_integer(2**66_037)
    coefficient_denominator = format_canonical_integer(3**42_017)
    entry_numerator = format_canonical_integer(7**24_048)
    entry_denominator = format_canonical_integer(5**28_072)

    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(
                entries=((_rational(entry_numerator, entry_denominator),),)
            ),
            polynomial=_rational_polynomial(
                (_rational(coefficient_numerator, coefficient_denominator), 1)
            ),
        )
    )


def test_linear_output_bounds_preserve_additive_cancellation() -> None:
    # [H] with f = t - H: both diagonal summands have magnitude H yet their
    # exact sum is zero, so magnitude-only addition charges 2H and rejects a
    # request whose digit work and exact value are small. Sign-preserving
    # reduction must admit the exact zero instead.
    height = "9" + "0" * 32_767
    request = MatrixPolynomialEvaluationRequest(
        matrix=RationalMatrix(entries=((_rational(height),),)),
        polynomial=_rational_polynomial(
            (_rational(1), 1), (_rational("-" + height), 0)
        ),
    )

    result = _evaluate(request)

    assert _fractions(result.value) == ((Fraction(0),),)
    assert result.polynomial_degree == 1
    assert result.matrix_multiplications == 1
    assert result.scalar_product_terms == 1


def test_linear_output_bounds_still_reject_uncancellable_additive_growth() -> None:
    # The additive twin without cancellation: f = t + H genuinely evaluates
    # to 2H, whose 32,769 digits exceed the canonical component cap, so the
    # reduced exact bound still rejects during request validation.
    height = "9" + "0" * 32_767
    _assert_admission_rejected(
        MatrixPolynomialEvaluationRequest(
            matrix=RationalMatrix(entries=((_rational(height),),)),
            polynomial=_rational_polynomial((_rational(1), 1), (_rational(height), 0)),
        )
    )


def test_request_schema_publishes_coupled_degree_order_work_bound() -> None:
    schema = MatrixPolynomialEvaluationRequest.model_json_schema()

    polynomial_description = schema["properties"]["polynomial"]["description"]
    assert "2 * degree * order^3" in polynomial_description
    assert f"{MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,}" in polynomial_description
    matrix_description = schema["properties"]["matrix"]["description"]
    assert "order 32" in matrix_description
