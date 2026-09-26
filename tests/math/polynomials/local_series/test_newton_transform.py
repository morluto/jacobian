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
from jacobian.math.polynomials.local_series.newton_transform import (
    NewtonTransformRequest,
    NewtonTransformResult,
    newton_transform,
)
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _window(
    coefficients: tuple[int | Fraction, ...], *, lower: int, precision: int
) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        place="FINITE",
        center=_q(0),
        valuation_lower=lower,
        precision=precision,
        coefficients=tuple(_q(value) for value in coefficients),
    )


def _polynomial(rows: tuple[tuple[int, TruncatedLaurentWindow], ...]) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_q(0),
        coefficients=tuple(
            LocalPolynomialCoefficient(y_degree=degree, series=series)
            for degree, series in rows
        ),
    )


def _terms(result: NewtonTransformResult) -> dict[tuple[int, int], Fraction]:
    terms: dict[tuple[int, int], Fraction] = {}
    for row in result.transformed_polynomial.coefficients:
        assert row.series is not None
        for offset, coefficient in enumerate(row.series.coefficients):
            value = coefficient.as_fraction()
            if value:
                terms[(row.series.valuation_lower + offset, row.y_degree)] = value
    return terms


def test_newton_transform_matches_direct_smooth_substitution() -> None:
    # Directly expand t^-2 F(t, t(1+z)) for F=y^2-t^2(1+t).
    source = _polynomial(
        (
            (0, _window((-1, -1), lower=2, precision=4)),
            (2, _window((1, 0, 0, 0), lower=0, precision=4)),
        )
    )
    result = newton_transform(
        NewtonTransformRequest(polynomial=source, edge_index=0, initial_root=_q(1))
    )
    assert _terms(result) == {
        (0, 1): Fraction(2),
        (0, 2): Fraction(1),
        (1, 0): Fraction(-1),
    }
    assert result.removed_valuation == 2
    assert result.characteristic.source == source
    assert NewtonTransformResult.model_validate_json(result.model_dump_json()) == result


def test_newton_transform_matches_direct_ramified_cusp_substitution() -> None:
    # Directly expand u^-6 F(u^2, u^3(c+z)) for F=y^2-t^3.
    source = _polynomial(
        (
            (0, _window((-1, 0, 0, 0), lower=3, precision=7)),
            (2, _window((1, 0, 0, 0, 0, 0, 0), lower=0, precision=7)),
        )
    )
    result = newton_transform(
        NewtonTransformRequest(polynomial=source, edge_index=0, initial_root=_q(1))
    )
    assert result.ramification_index == 2
    assert result.ordinate_power == 3
    assert result.removed_valuation == 6
    assert _terms(result) == {
        (0, 1): Fraction(2),
        (0, 2): Fraction(1),
    }
    assert NewtonTransformResult.model_validate_json(result.model_dump_json()) == result


def test_newton_transform_rejects_invalid_and_multiple_roots() -> None:
    simple = _polynomial(
        (
            (0, _window((-1, -1), lower=2, precision=4)),
            (2, _window((1, 0, 0, 0), lower=0, precision=4)),
        )
    )
    with pytest.raises(OperationDomainValidationError) as invalid_root:
        newton_transform(
            NewtonTransformRequest(
                polynomial=simple, edge_index=0, initial_root=_q(2)
            )
        )
    assert invalid_root.value.errors()[0]["type"] == (
        "local_series.newton_transform_root_not_on_edge"
    )

    multiple = _polynomial(
        (
            (0, _window((1, 0, 0), lower=2, precision=5)),
            (1, _window((-2, 0, 0, 0, 0), lower=1, precision=6)),
            (2, _window((1, 0, 0, 0, 0), lower=0, precision=5)),
        )
    )
    with pytest.raises(OperationDomainValidationError) as multiple_root:
        newton_transform(
            NewtonTransformRequest(
                polynomial=multiple, edge_index=0, initial_root=_q(1)
            )
        )
    assert multiple_root.value.errors()[0]["type"] == (
        "local_series.newton_transform_multiple_root"
    )


def test_newton_transform_rejects_noncanonical_native_root_before_conversion() -> None:
    source = _polynomial(
        (
            (0, _window((-1, -1), lower=2, precision=4)),
            (2, _window((1, 0, 0, 0), lower=0, precision=4)),
        )
    )
    request = NewtonTransformRequest.model_construct(
        polynomial=source,
        edge_index=0,
        initial_root=CanonicalRational.model_construct(num=1, den=0),
    )
    with pytest.raises(OperationDomainValidationError) as invalid_root:
        newton_transform(request)
    assert invalid_root.value.errors()[0]["type"] == (
        "local_series.newton_transform_root_canonical"
    )


def test_newton_transform_pre_admits_expanded_output_slots() -> None:
    precision = 4_000
    source = _polynomial(
        (
            (
                0,
                _window(
                    (-1, -1, *((0,) * (precision - 4))),
                    lower=2,
                    precision=precision,
                ),
            ),
            (
                2,
                _window((1, *((0,) * (precision - 1))), lower=0, precision=precision),
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as oversized:
        newton_transform(
            NewtonTransformRequest(polynomial=source, edge_index=0, initial_root=_q(1))
        )
    assert oversized.value.errors()[0]["type"] == (
        "local_series.newton_transform_output_slots_bound"
    )


def test_newton_transform_is_published_once() -> None:
    matches = [
        tool
        for tool in TOOLS
        if tool.operation_id == "local_series.polynomial.newton_transform.compute"
    ]
    assert len(matches) == 1
