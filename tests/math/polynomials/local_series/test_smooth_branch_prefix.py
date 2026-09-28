from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series._tools import TOOLS
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
)
from jacobian.math.polynomials.local_series.smooth_branch import (
    SmoothBranchPrefixRequest,
    SmoothBranchPrefixResult,
    smooth_branch_prefix,
)
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _series(
    coefficients: tuple[int | Fraction, ...], precision: int
) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        place="FINITE",
        center=_rational(0),
        valuation_lower=0,
        precision=precision,
        coefficients=tuple(_rational(value) for value in coefficients),
    )


def _sqrt_one_plus_t(precision: int) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=0,
                series=_series((-1, -1, *((0,) * (precision - 2))), precision),
            ),
            LocalPolynomialCoefficient(
                y_degree=2,
                series=_series((1, *((0,) * (precision - 1))), precision),
            ),
        ),
    )


def _convolution(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...], precision: int
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (left[i] * right[degree - i] for i in range(degree + 1)),
            Fraction(0),
        )
        for degree in range(precision)
    )


def test_smooth_branch_prefix_matches_exact_square_identity() -> None:
    source = _sqrt_one_plus_t(5)

    result = smooth_branch_prefix(
        SmoothBranchPrefixRequest(
            polynomial=source,
            initial_root=_rational(1),
            precision=5,
        )
    )

    coefficients = tuple(value.as_fraction() for value in result.series.coefficients)
    assert coefficients == (
        Fraction(1),
        Fraction(1, 2),
        Fraction(-1, 8),
        Fraction(1, 16),
        Fraction(-5, 128),
    )
    square = _convolution(coefficients, coefficients, 5)
    assert square == (Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    assert result.residual_precision == result.series.precision == 5
    assert result.source == source
    assert (
        SmoothBranchPrefixResult.model_validate_json(result.model_dump_json()) == result
    )


def test_smooth_branch_prefix_accepts_exact_requested_source_precision() -> None:
    result = smooth_branch_prefix(
        SmoothBranchPrefixRequest(
            polynomial=_sqrt_one_plus_t(3),
            initial_root=_rational(1),
            precision=3,
        )
    )

    assert tuple(value.as_fraction() for value in result.series.coefficients) == (
        Fraction(1),
        Fraction(1, 2),
        Fraction(-1, 8),
    )


def test_smooth_branch_prefix_accepts_maximum_precision() -> None:
    precision = 32
    source = LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=0,
                series=_series((-1, -1, *((0,) * (precision - 2))), precision),
            ),
            LocalPolynomialCoefficient(
                y_degree=1,
                series=_series((1, *((0,) * (precision - 1))), precision),
            ),
        ),
    )

    result = smooth_branch_prefix(
        SmoothBranchPrefixRequest(
            polynomial=source,
            initial_root=_rational(1),
            precision=precision,
        )
    )

    assert result.series.precision == precision
    assert tuple(value.as_fraction() for value in result.series.coefficients) == (
        Fraction(1),
        Fraction(1),
        *((Fraction(0),) * (precision - 2)),
    )


def test_smooth_branch_prefix_requires_source_through_requested_order() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_prefix(
            SmoothBranchPrefixRequest(
                polynomial=_sqrt_one_plus_t(4),
                initial_root=_rational(1),
                precision=5,
            )
        )

    assert (
        error.value.errors()[0]["type"] == "local_series.smooth_branch.prefix_precision"
    )


def test_smooth_branch_prefix_rejects_multiple_initial_root() -> None:
    source = LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(y_degree=0, series=_series((0, 0, 0), 3)),
            LocalPolynomialCoefficient(y_degree=2, series=_series((1, 0, 0), 3)),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_prefix(
            SmoothBranchPrefixRequest(
                polynomial=source,
                initial_root=_rational(0),
                precision=3,
            )
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.not_simple"


def test_smooth_branch_prefix_admits_height_before_expansion() -> None:
    precision = 20
    small_derivative = Fraction(1, 10**255)
    source = LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=0,
                series=_series((0, -1, *((0,) * (precision - 2))), precision),
            ),
            LocalPolynomialCoefficient(
                y_degree=1,
                series=_series(
                    (small_derivative, *((0,) * (precision - 1))), precision
                ),
            ),
            LocalPolynomialCoefficient(
                y_degree=2,
                series=_series((0, 1, *((0,) * (precision - 2))), precision),
            ),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        smooth_branch_prefix(
            SmoothBranchPrefixRequest(
                polynomial=source,
                initial_root=_rational(0),
                precision=precision,
            )
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.prefix_growth"


def test_smooth_branch_prefix_is_published_as_distinct_higher_order_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "local_series.polynomial.smooth_branch_prefix.compute"
    )

    assert tool.request_type is SmoothBranchPrefixRequest
    assert tool.result_type is SmoothBranchPrefixResult
