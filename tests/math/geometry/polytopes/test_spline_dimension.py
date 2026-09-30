from fractions import Fraction
from threading import Event
from typing import Any

import pytest
import sympy as sp
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_checkpoint,
)
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    SplineDimensionRequest,
    SplineEvaluationRequest,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    spline_dimension,
    spline_evaluate,
    spline_space,
)


def _box(
    x0: int | Fraction,
    x1: int | Fraction,
    y0: int | Fraction,
    y1: int | Fraction,
    prefix: str,
) -> RationalVPolytope:
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=tuple(
                    CanonicalRational.from_fraction(Fraction(value)) for value in point
                ),
            )
            for index, point in enumerate(points)
        ),
    )


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=(CanonicalRational(num=value, den=1),),
            )
            for index, value in enumerate((left, right))
        ),
    )


def _rank(rows: list[list[Fraction]]) -> int:
    """Small independent exact row-reduction oracle for the interval fixture."""
    if not rows:
        return 0
    matrix = [row[:] for row in rows]
    pivot_row = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return pivot_row


def _interval_continuity_matrix(degree: int, smoothness: int) -> list[list[Fraction]]:
    """Match derivatives at x=1 for two independent degree-d polynomials."""
    rows = []
    for derivative in range(smoothness + 1):
        row = [Fraction(0) for _ in range(2 * (degree + 1))]
        for exponent in range(derivative, degree + 1):
            coefficient = Fraction(
                1,
            )
            for factor in range(derivative):
                coefficient *= exponent - factor
            row[exponent] = coefficient
            row[degree + 1 + exponent] = -coefficient
        rows.append(row)
    return rows


def test_spline_dimension_matches_interval_derivative_oracle_and_full_space() -> None:
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    degree, smoothness = 4, 2

    result = spline_dimension(
        SplineDimensionRequest(
            complex=complex_value, degree=degree, smoothness=smoothness
        )
    )
    full = spline_space(complex_value, degree, smoothness)
    oracle_rank = _rank(_interval_continuity_matrix(degree, smoothness))

    assert result.rank == oracle_rank == full.rank
    assert result.nullity == 2 * (degree + 1) - oracle_rank == full.nullity
    assert result.compatibility_matrix == full.compatibility_matrix
    assert result.coefficient_axis == full.coefficient_axis


def test_dimension_output_bound_is_conservative_at_its_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    request = SplineDimensionRequest(complex=complex_value, degree=3, smoothness=0)
    result = spline_dimension(request)
    rows = tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.compatibility_matrix.entries
    )
    _, width, row_bound, _ = spline_kernel._admit_spline_dimension(complex_value, 3, 0)
    _, _, estimate = spline_kernel._admit_spline_dimension_height(
        complex_value, 3, 0, width, row_bound
    )
    stored_digits = sum(
        decimal_digit_width(value.numerator) + decimal_digit_width(value.denominator)
        for row in rows
        for value in row
    )

    assert rows and stored_digits > 0
    assert estimate >= stored_digits
    monkeypatch.setattr(spline_kernel, "MAX_SPLINE_DIMENSION_OUTPUT_DIGITS", estimate)
    assert spline_dimension(request).nullity == result.nullity
    monkeypatch.setattr(
        spline_kernel, "MAX_SPLINE_DIMENSION_OUTPUT_DIGITS", estimate - 1
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        spline_dimension(request)
    assert (
        error.value.errors()[0]["type"] == "polytopal_complex.spline_dimension_output"
    )


def test_dimension_admits_matrix_when_full_basis_output_exceeds_its_bound() -> None:
    cells = tuple(_box(index, index + 1, 0, 1, f"c{index}-") for index in range(10))
    complex_value = polytopal_complex_closure(cells)
    request = SplineDimensionRequest(complex=complex_value, degree=12, smoothness=0)

    with pytest.raises(OperationResourceAdmissionError, match="exact result"):
        spline_space(complex_value, 12, 0)

    result = spline_dimension(request)

    # Ten cellwise degree-12 polynomials in two variables have 910 coordinates.
    # Each of the nine vertical interfaces equates 13 independent y-polynomial
    # coefficients, so the dimension is 910 - 9*13 = 793.
    assert result.compatibility_matrix.row_count == 117
    assert result.compatibility_matrix.column_count == 910
    assert result.rank == 117
    assert result.nullity == 793


def test_native_spline_entry_points_reject_forged_requests_with_typed_errors() -> None:
    with pytest.raises(OperationDomainValidationError, match="dimension request"):
        spline_dimension(SplineDimensionRequest.model_construct())
    with pytest.raises(OperationDomainValidationError, match="evaluation request"):
        spline_evaluate(SplineEvaluationRequest.model_construct())


def test_spline_evaluation_admits_scalars_whose_exact_result_is_at_the_limit() -> None:
    """A basis coefficient's own width is not the evaluation's output width.

    The preflight charged each coefficient's numerator *and* denominator width
    to the numerator, added a unit basis entry's width, and compared the running
    total of every term against the 32,768-digit per-component limit. A
    32,768-digit carrier-valid scalar was therefore refused on a 32,770-digit
    estimate even where the exact value is that scalar unchanged.
    """
    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.math.geometry.polytopes.complexes._models import (
        ComplexPoint,
        SplineEvaluationRequest,
    )
    from jacobian.math.geometry.polytopes.complexes.operations import spline_evaluate

    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    # degree 2 on one interval: three basis rows over the monomials (x^2, x, 1)
    spline = spline_space(complex_value, 2, 0)
    assert spline.nullity == 3
    point = ComplexPoint(coordinates=(CanonicalRational(num=1, den=2),))
    limit = MAX_CANONICAL_RATIONAL_DIGITS
    # 2**108850 has exactly 32,768 decimal digits, the carrier maximum
    at_limit = 2**108850
    assert decimal_digit_width(at_limit) == limit

    def evaluate(coefficient: CanonicalRational) -> CanonicalRational:
        request = SplineEvaluationRequest(
            complex=complex_value,
            degree=2,
            smoothness=0,
            basis_coefficients=(CanonicalRational(num=0, den=1),) * 2 + (coefficient,),
            point=point,
        )
        value = spline_evaluate(request).value
        assert value is not None
        return value

    # the x^0 basis row selects the constant, so this is the scalar itself
    assert evaluate(CanonicalRational(num=at_limit, den=1)).num == at_limit
    assert evaluate(CanonicalRational(num=1, den=at_limit)).den == at_limit

    # a genuinely oversized exact result is still refused: the point's own
    # denominator multiplies the coefficient's into a 65,537-digit rational
    oversized_point = ComplexPoint(
        coordinates=(CanonicalRational(num=1, den=at_limit),)
    )
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        spline_evaluate(
            SplineEvaluationRequest(
                complex=complex_value,
                degree=2,
                smoothness=0,
                basis_coefficients=(
                    CanonicalRational(num=0, den=1),
                    CanonicalRational(num=1, den=at_limit),
                    CanonicalRational(num=0, den=1),
                ),
                point=oversized_point,
            )
        )


def test_spline_evaluation_cancels_factors_before_summing() -> None:
    """A coefficient of 1/N against a point of N/2 evaluates to an exact 1/2.

    The preflight added factor widths without reducing, so this was refused
    while the exact value is one digit wide.
    """
    from jacobian.math.geometry.polytopes.complexes._models import (
        ComplexPoint,
        SplineEvaluationRequest,
    )
    from jacobian.math.geometry.polytopes.complexes.operations import spline_evaluate

    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    big = 2**108850
    # basis row 1 selects the x monomial, so N against the point 1/N is 1
    point = ComplexPoint(coordinates=(CanonicalRational(num=1, den=big),))
    request = SplineEvaluationRequest(
        complex=complex_value,
        degree=2,
        smoothness=0,
        basis_coefficients=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=big, den=1),
            CanonicalRational(num=0, den=1),
        ),
        point=point,
    )
    value = spline_evaluate(request).value
    assert value is not None
    assert value.num == 1
    assert value.den == 1


@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("degree", (2, 3))
def test_spline_evaluation_cancels_each_factor_only_once(
    sign: int, degree: int
) -> None:
    """N * (1/N)^d is 1/N^(d-1), even with repeated denominator factors."""
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    at_limit = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    request = SplineEvaluationRequest(
        complex=complex_value,
        degree=degree,
        smoothness=0,
        basis_coefficients=(CanonicalRational(num=sign * at_limit, den=1),)
        + (CanonicalRational(num=0, den=1),) * degree,
        point=ComplexPoint(coordinates=(CanonicalRational(num=1, den=at_limit),)),
    )

    if degree == 2:
        assert spline_evaluate(request).value == CanonicalRational(
            num=sign, den=at_limit
        )
    else:
        # Squaring the carrier-sized denominator exceeds the output envelope.
        with pytest.raises(OperationResourceAdmissionError) as error:
            spline_evaluate(request)
        assert error.value.errors()[0]["type"] == (
            "polytopal_complex.spline_evaluation_growth"
        )


@pytest.mark.parametrize("constant_kind", ("zero", "integer", "rational"))
def test_spline_evaluation_omits_zero_monomials(constant_kind: str) -> None:
    """At x=0 only the constant survives, regardless of other coefficients."""
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    at_limit = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    constant = {
        "zero": CanonicalRational(num=0, den=1),
        "integer": CanonicalRational(num=at_limit, den=1),
        "rational": CanonicalRational(num=1, den=at_limit),
    }[constant_kind]
    request = SplineEvaluationRequest(
        complex=complex_value,
        degree=2,
        smoothness=0,
        basis_coefficients=(CanonicalRational(num=1, den=at_limit),) * 2 + (constant,),
        point=ComplexPoint(coordinates=(CanonicalRational(num=0, den=1),)),
    )

    assert spline_evaluate(request).value == constant


@pytest.mark.parametrize(
    "coefficients",
    [
        (2, -3, 5, 7),
        (0, 2, -3, 5),
        (0, 0, -7, 0),
        (
            sp.Rational(2, 7),
            sp.Rational(-3, 11),
            sp.Rational(5, 13),
            sp.Rational(7, 17),
        ),
    ],
)
@pytest.mark.parametrize("degree,smoothness", [(0, 0), (3, 0), (4, 2), (5, 3), (3, 4)])
def test_source_remainder_bound_and_defining_identity(
    coefficients: tuple[Any, ...], degree: int, smoothness: int
) -> None:
    x, y, z = symbols = sp.symbols("x y z")
    ell = sp.Poly(
        sum(c * v for c, v in zip(coefficients, (x, y, z, 1), strict=True)),
        *symbols,
        domain=sp.QQ,
    )
    leading, bounds, _ = spline_kernel._facet_remainder_bounds(ell, degree, smoothness)
    assert spline_kernel._facet_remainder_bounds(
        ell.mul_ground(sp.Rational(-13, 19)), degree, smoothness
    ) == (
        leading,
        bounds,
        spline_kernel._facet_remainder_bounds(ell, degree, smoothness)[2],
    )
    monomials = spline_kernel._monomials(3, degree)
    remainders = spline_kernel._spline_facet_remainders(
        ell, monomials, degree, smoothness
    )
    for exponents, remainder_coefficients in zip(monomials, remainders, strict=True):
        remainder = sp.Poly.from_dict(remainder_coefficients, symbols, domain=sp.QQ)
        monomial = sp.Poly(
            sp.prod(v**e for v, e in zip(symbols, exponents, strict=True)),
            *symbols,
            domain=sp.QQ,
        )
        numerator, denominator = bounds[exponents[leading]]
        assert all(
            denominator % int(c.q) == 0 and abs(c * denominator) <= numerator
            for c in remainder.coeffs()
        )
        # Independent Taylor remainder: its derivatives through order r
        # agree with the monomial at the hyperplane, and x-degree is < r+1.
        root = sp.solve(ell.as_expr(), symbols[leading])[0]
        for derivative in range(smoothness + 1):
            difference = sp.diff(
                (monomial - remainder).as_expr(), symbols[leading], derivative
            )
            assert sp.expand(difference.subs(symbols[leading], root)) == 0
        assert remainder.degree(symbols[leading]) <= smoothness


@pytest.mark.parametrize("horizontal", [False, True])
def test_dimension_axis_permutation_preserves_continuity(horizontal: bool) -> None:
    cells = (
        (_box(0, 1, 0, 1, "a"), _box(0, 1, 1, 2, "b"))
        if horizontal
        else (_box(0, 1, 0, 1, "a"), _box(1, 2, 0, 1, "b"))
    )
    complex_value = polytopal_complex_closure(cells)
    result = spline_dimension(
        SplineDimensionRequest(complex=complex_value, degree=3, smoothness=0)
    )
    assert result.rank == 4 and result.nullity == 16
    assert (
        spline_space(complex_value, 3, 0).compatibility_matrix
        == result.compatibility_matrix
    )


def test_multiple_rational_facets_aggregate_growth_and_reduction_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = Fraction(1, 3), Fraction(2, 5)
    complex_value = polytopal_complex_closure(
        (
            _box(0, x, 0, y, "a"),
            _box(x, 1, 0, y, "b"),
            _box(0, x, y, 1, "c"),
            _box(x, 1, y, 1, "d"),
        )
    )
    powers = 0
    original = spline_kernel._spline_linear_powers

    def counted_powers(terms: Any, dimension: int, degree: int) -> Any:
        nonlocal powers
        powers += 1
        return original(terms, dimension, degree)

    def no_division(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "spline construction must use the closed form, not polynomial division"
        )

    monkeypatch.setattr(spline_kernel, "_spline_linear_powers", counted_powers)
    monkeypatch.setattr(sp.Poly, "div", no_division)
    result = spline_dimension(
        SplineDimensionRequest(complex=complex_value, degree=2, smoothness=0)
    )
    # Four quadratic restrictions have one cycle relation at the central point.
    assert result.rank == 4 * 3 - 1 and result.nullity == 13
    assert_charged_work_parity(
        charged={"facet_power_caches": 4},
        executed={"facet_power_caches": powers},
    )
    assert powers == 4
    rows = [
        [entry.as_fraction() for entry in row]
        for row in result.compatibility_matrix.entries
    ]
    assert _rank(rows) == result.rank
    _, width, row_bound, _ = spline_kernel._admit_spline_dimension(complex_value, 2, 0)
    _, _, output_bound = spline_kernel._admit_spline_dimension_height(
        complex_value, 2, 0, width, row_bound
    )
    assert (
        sum(
            decimal_digit_width(v.numerator) + decimal_digit_width(v.denominator)
            for row in rows
            for v in row
        )
        <= output_bound
    )


def test_inactive_first_axis_remainder_matches_original_ambient_order() -> None:
    # Old Poly(x*y, x,y).div(Poly(y-1, x,y)) left x*y unreduced.
    # The shared row-bound check then rejected the horizontal two-cell space.
    x, y = sp.symbols("x y")
    remainder = spline_kernel._spline_facet_remainders(
        sp.Poly(y - 1, x, y, domain=sp.QQ),
        ((1, 1),),
        2,
        0,
    )
    assert remainder == ({(1, 0): Fraction(1)},)


@pytest.mark.parametrize(
    "phase",
    [
        "source coefficient bounds",
        "global column admission",
        "cache storage admission",
        "facet power convolution",
        "facet denominator powers",
        "facet Taylor coefficients",
        "remainder coefficient shifts",
        "compatibility matrix assembly",
        "canonical matrix construction",
        "rank column clearing",
        "integer matrix construction",
    ],
)
def test_spline_cancels_inside_bounded_phases(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    cancelled = Event()
    observed = 0

    def cancel_during_work(stage: str) -> None:
        nonlocal observed
        if phase in stage:
            observed += 1
            if observed == 2:
                cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(spline_kernel, "request_checkpoint", cancel_during_work)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError),
    ):
        spline_dimension(
            SplineDimensionRequest(complex=complex_value, degree=4, smoothness=2)
        )
    assert observed == 2
