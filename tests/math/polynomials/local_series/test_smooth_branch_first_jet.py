from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
)
from jacobian.math.polynomials.local_series.smooth_branch import (
    SmoothBranchFirstJetRequest,
    SmoothBranchFirstJetResult,
    smooth_branch_first_jet,
)
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _series(
    *coefficients: int, lower: int = 0, precision: int = 2
) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        place="FINITE",
        center=_rational(0),
        valuation_lower=lower,
        precision=precision,
        coefficients=tuple(_rational(value) for value in coefficients),
    )


def _polynomial(
    rows: tuple[tuple[int, TruncatedLaurentWindow | None], ...],
) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=tuple(
            LocalPolynomialCoefficient(y_degree=degree, series=series)
            for degree, series in rows
        ),
    )


def test_smooth_branch_first_jet_matches_independent_factorized_oracle() -> None:
    # Modulo t^2 this is (y - (2 + 3t)) (y + 1 + 7t).
    source = _polynomial(
        (
            (0, _series(-2, -17, 11, precision=3)),
            (1, _series(-1, 4, -31, precision=3)),
            (2, _series(1, 0, 5, precision=3)),
        )
    )

    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(2))
    )

    assert result.series.coefficients == (_rational(2), _rational(3))
    assert result.series.precision == result.residual_precision == 2
    # Independent coefficient identities at c=2: F(0,c)=0,
    # F_y(0,c)=3, and F_t(0,c)=-9, so the unique slope is 3.
    assert 2**2 - 2 - 2 == 0
    assert 2 * 2 - 1 == 3
    assert -17 + 4 * 2 == -9
    assert Fraction(9, 3) == result.series.coefficients[1].as_fraction()
    decoded = SmoothBranchFirstJetResult.model_validate_json(result.model_dump_json())
    assert decoded == result

    different_tail = _polynomial(
        (
            (0, _series(-2, -17, -999, precision=3)),
            (1, _series(-1, 4, 123, precision=3)),
            (2, _series(1, 0, 8, precision=3)),
        )
    )
    other_result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(
            polynomial=different_tail, initial_root=_rational(2)
        )
    )
    assert other_result.series == result.series


def test_smooth_branch_preserves_zero_jet_with_zero_finite_coefficient_prefix() -> None:
    # The degree-two coefficient has a finite, all-zero prefix through t^1.
    # It contributes no obstruction to the valid constant branch y=2.
    source = _polynomial(
        (
            (0, _series(-2, 0)),
            (1, _series(1, 0)),
            (2, _series(0, 0)),
        )
    )

    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(2))
    )

    assert result.series.coefficients == (_rational(2), _rational(0))


def test_smooth_branch_rejects_a_value_that_is_not_a_root() -> None:
    source = _polynomial(((0, _series(-2, 0)), (1, _series(1, 0))))

    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(3))
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.not_root"


def test_smooth_branch_rejects_a_multiple_root() -> None:
    source = _polynomial(((0, None), (2, _series(1, 0))))

    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.not_simple"


def test_smooth_branch_requires_coefficient_prefix_through_t() -> None:
    source = _polynomial(
        (
            (0, _series(-2, precision=1)),
            (1, _series(1, precision=1)),
        )
    )

    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(2))
        )

    assert (
        error.value.errors()[0]["type"] == "local_series.smooth_branch.input_precision"
    )


def test_smooth_branch_rejects_principal_parts() -> None:
    source = _polynomial(
        (
            (0, _series(1, 0, 0, lower=-1, precision=2)),
            (1, _series(1, 0)),
        )
    )

    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.pole"


def test_smooth_branch_rejects_negative_power_even_when_prefix_reaches_t() -> None:
    source = _polynomial(
        (
            (0, _series(1, 0, 0, lower=-1, precision=2)),
            (1, _series(1, 0)),
        )
    )

    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
        )

    assert error.value.errors()[0]["type"] == "local_series.smooth_branch.pole"


def test_smooth_branch_rejects_degree_outside_slice() -> None:
    source = _polynomial(((17, _series(1, 0)),))

    with pytest.raises(OperationResourceAdmissionError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
        )

    assert "degree at most 16" in str(error.value)


def test_smooth_branch_accepts_maximum_degree() -> None:
    source = _polynomial(((0, _series(-1, 0)), (16, _series(1, 0))))

    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
    )

    assert result.series.coefficients == (_rational(1), _rational(0))


def test_smooth_branch_accepts_exact_source_slot_bound() -> None:
    zeros = (0,) * 254
    source = _polynomial(
        (
            (0, _series(-1, 0, *zeros, precision=256)),
            (1, _series(1, 0, *zeros, precision=256)),
        )
    )

    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
    )

    assert result.series.coefficients == (_rational(1), _rational(0))


def test_smooth_branch_rejects_rational_growth_before_evaluation() -> None:
    large = 10**255
    source = _polynomial(tuple((degree, _series(large, 0)) for degree in range(17)))

    with pytest.raises(OperationResourceAdmissionError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
        )

    assert "intermediate bound" in str(error.value)


def test_zero_root_nonzero_slope_uses_canonical_laurent_valuation() -> None:
    source = _polynomial(((0, _series(0, -1)), (1, _series(1, 0))))

    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(0))
    )

    assert result.series.valuation_lower == 1
    assert result.series.coefficients == (_rational(1),)
    decoded = SmoothBranchFirstJetResult.model_validate_json(result.model_dump_json())
    assert decoded == result
