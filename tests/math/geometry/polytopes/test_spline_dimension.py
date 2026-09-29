from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
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
    SplineDimensionRequest,
    SplineEvaluationRequest,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    spline_dimension,
    spline_evaluate,
    spline_space,
)


def _box(x0: int, x1: int, y0: int, y1: int, prefix: str) -> RationalVPolytope:
    points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=tuple(
                    CanonicalRational(num=value, den=1) for value in point
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


def test_spline_dimension_matches_interval_derivative_oracle_and_full_space():
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


def test_dimension_output_bound_is_conservative_at_its_boundary(monkeypatch):
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    request = SplineDimensionRequest(complex=complex_value, degree=3, smoothness=0)
    result = spline_dimension(request)
    rows = tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.compatibility_matrix.entries
    )
    max_entry_digits = max(
        (
            max(
                decimal_digit_width(value.numerator),
                decimal_digit_width(value.denominator),
            )
            for row in rows
            for value in row
        ),
        default=1,
    )
    estimate = spline_kernel._spline_dimension_output_digit_bound(
        rows, result.compatibility_matrix.column_count, max_entry_digits
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
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        spline_dimension(request)


def test_dimension_admits_matrix_when_full_basis_output_exceeds_its_bound():
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


def test_native_spline_entry_points_reject_forged_requests_with_typed_errors():
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
    oversized_point = ComplexPoint(coordinates=(CanonicalRational(num=1, den=at_limit),))
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
