from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
    local_polynomial_newton_polygon,
)
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _series(lower: int, *coefficients: tuple[int, int]) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        place="FINITE",
        center={"num": 0, "den": 1},
        valuation_lower=lower,
        precision=lower + len(coefficients),
        coefficients=tuple({"num": n, "den": d} for n, d in coefficients),
    )


def _polynomial(
    rows: list[tuple[int, TruncatedLaurentWindow | None]],
) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center={"num": 0, "den": 1},
        coefficients=tuple(
            LocalPolynomialCoefficient(y_degree=degree, series=series)
            for degree, series in rows
        ),
    )


def test_newton_polygon_exact_edges_and_collinear_source_degree() -> None:
    result = local_polynomial_newton_polygon(
        _polynomial(
            [
                (0, _series(4, (1, 1))),
                (1, _series(2, (3, 2))),
                (2, _series(0, (1, 1))),
                (3, _series(1, (2, 1))),
                (4, _series(2, (1, 1))),
            ]
        )
    )
    assert [(p.y_degree, p.valuation) for p in result.vertices] == [
        (0, 4),
        (2, 0),
        (4, 2),
    ]
    assert [edge.slope.as_fraction() for edge in result.edges] == [
        Fraction(-2),
        Fraction(1),
    ]
    assert result.edges[0].source_y_degrees == (0, 1, 2)
    assert result.edges[1].source_y_degrees == (2, 3, 4)
    assert result.edges[0].horizontal_length == 2


def test_exact_zero_and_empty_polynomial() -> None:
    result = local_polynomial_newton_polygon(_polynomial([(0, None), (2, None)]))
    assert result.coefficient_valuations == ((0, None), (2, None))
    assert result.points == result.vertices == result.edges == ()
    empty = local_polynomial_newton_polygon(_polynomial([]))
    assert empty.vertices == ()


def test_all_zero_prefix_does_not_claim_infinite_valuation() -> None:
    source = _polynomial([(0, _series(0, (0, 1), (0, 1)))])
    with pytest.raises(OperationDomainValidationError) as error:
        local_polynomial_newton_polygon(source)
    assert error.value.errors()[0]["type"] == "local_series.newton_valuation_unknown"


def test_mismatched_local_parent_is_rejected() -> None:
    with pytest.raises(ValueError, match="share the declared local parent"):
        _polynomial(
            [
                (
                    0,
                    TruncatedLaurentWindow(
                        variable="s",
                        place="FINITE",
                        center={"num": 0, "den": 1},
                        valuation_lower=0,
                        precision=1,
                        coefficients=({"num": 1, "den": 1},),
                    ),
                )
            ]
        )


def test_singleton_and_zero_row_preserve_exact_valuation() -> None:
    result = local_polynomial_newton_polygon(
        _polynomial([(0, None), (5, _series(-2, (0, 1), (7, 3)))])
    )
    assert result.coefficient_valuations == ((0, None), (5, -1))
    assert [(p.y_degree, p.valuation) for p in result.vertices] == [(5, -1)]
    assert result.edges == ()
