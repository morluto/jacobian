import json
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    DFinitePowerSeries,
    DifferentialOreOperator,
)
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import differential_series_construct
from jacobian.math.polynomials.values import RationalFunction


def _rf(terms: list[tuple[int, int]], *, variable: str = "x") -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": [variable],
            "numerator": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": [degree],
                    }
                    for coefficient, degree in terms
                ]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
        }
    )


def _sinh_operator() -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "terms": [
                {"order": 0, "coefficient": _rf([(-1, 0)]).model_dump()},
                {"order": 2, "coefficient": _rf([(1, 0)]).model_dump()},
            ]
        }
    )


def test_sinh_ode_carrier_matches_independent_taylor_recurrence() -> None:
    carrier = differential_series_construct(
        _sinh_operator(),
        FiniteRationalSequence.model_validate({"values": [0, 1]}),
    )

    # Independent coefficient comparison in y''-y=0 gives
    # (n+2)(n+1)a_(n+2)=a_n; this oracle does not call the Ore kernel.
    coefficients = [Fraction(0), Fraction(1)]
    for n in range(8):
        coefficients.append(coefficients[n] / ((n + 2) * (n + 1)))
    assert coefficients == [
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(1, 6),
        Fraction(0),
        Fraction(1, 120),
        Fraction(0),
        Fraction(1, 5040),
        Fraction(0),
        Fraction(1, 362880),
    ]
    assert carrier.center == 0
    assert carrier.operator == _sinh_operator()
    assert [value.as_fraction() for value in carrier.initial_derivatives.values] == [
        Fraction(0),
        Fraction(1),
    ]
    assert DFinitePowerSeries.model_validate_json(carrier.model_dump_json()) == carrier


def test_catalog_operation_returns_the_dfinite_target_value() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "holonomic.differential_series.construct"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, DFinitePowerSeries)
    assert result.operator == _sinh_operator()


def test_singular_ode_center_is_not_misrepresented_as_unique_series() -> None:
    operator = DifferentialOreOperator.model_validate(
        {
            "terms": [
                {"order": 0, "coefficient": _rf([(1, 0)]).model_dump()},
                {"order": 2, "coefficient": _rf([(1, 1)]).model_dump()},
            ]
        }
    )
    with pytest.raises(OperationDomainValidationError, match="ordinary center"):
        differential_series_construct(operator, {"values": [0, 0]})


def test_coefficient_pole_at_center_is_rejected() -> None:
    pole = RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [1]}]
            },
        }
    )
    with pytest.raises(OperationDomainValidationError, match="ordinary center"):
        differential_series_construct(
            DifferentialOreOperator.model_validate(
                {"terms": [{"order": 1, "coefficient": pole.model_dump()}]}
            ),
            {"values": [0]},
        )
