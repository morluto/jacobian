import json
from fractions import Fraction
from math import factorial

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import (
    differential_operator_to_coefficient_recurrence,
)


def _rf(terms: list[tuple[int, int]]) -> dict:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [
                {"coefficient": {"num": value, "den": 1}, "exponents": [degree]}
                for degree, value in terms
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }


def _operator(*terms: tuple[int, list[tuple[int, int]]]) -> dict:
    return {
        "variable": "x",
        "terms": [
            {"order": order, "coefficient": _rf(polynomial)}
            for order, polynomial in terms
        ],
    }


def _polynomial(coefficient) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in coefficient.numerator.terms
    }


def test_exponential_equation_yields_factorial_recurrence() -> None:
    operator = DifferentialOreOperator.model_validate(
        _operator((0, [(0, -1)]), (1, [(0, 1)]))
    )
    result = differential_operator_to_coefficient_recurrence(operator)
    assert result.valid_from == 0
    assert [
        (term.shift, _polynomial(term.coefficient)) for term in result.recurrence
    ] == [
        (0, {0: Fraction(-1)}),
        (1, {0: Fraction(1), 1: Fraction(1)}),
    ]
    assert result.boundary_rows == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result
    # Exact independent solution a_n=1/n! satisfies every emitted row.
    for n in range(result.valid_from, 24):
        lhs = sum(
            sum(
                value * n**degree
                for degree, value in _polynomial(term.coefficient).items()
            )
            / factorial(n + term.shift)
            for term in result.recurrence
        )
        assert lhs == 0


def test_sinh_equation_has_exact_second_order_recurrence() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (2, [(0, 1)]))
    )
    recurrence = {
        term.shift: _polynomial(term.coefficient) for term in result.recurrence
    }
    assert recurrence == {
        0: {0: Fraction(-1)},
        2: {0: Fraction(2), 1: Fraction(3), 2: Fraction(1)},
    }
    # The independent exact solution y=e^x has coefficients a_n=1/n!.
    coefficients = [Fraction(1, factorial(index)) for index in range(40)]
    for n in range(18):
        lhs = -coefficients[n] + (n + 2) * (n + 1) * coefficients[n + 2]
        assert lhs == 0


def test_low_degree_boundary_row_is_retained() -> None:
    # x*y' - y = 0 has the exceptional row -a_0=0 before (n-1)a_n=0.
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (1, [(1, 1)]))
    )
    assert result.valid_from == 1
    assert len(result.boundary_rows) == 1
    assert result.boundary_rows[0].degree == 0
    assert [
        (term.index, term.coefficient.as_fraction())
        for term in result.boundary_rows[0].terms
    ] == [(0, Fraction(-1))]
    assert result.recurrence[0].shift == 0
    assert _polynomial(result.recurrence[0].coefficient) == {
        0: Fraction(-1),
        1: Fraction(1),
    }


def test_rational_function_ode_coefficients_are_rejected() -> None:
    rational = _rf([(0, 1)])
    rational["denominator"] = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        differential_operator_to_coefficient_recurrence(
            {"variable": "x", "terms": [{"order": 0, "coefficient": rational}]}
        )


def test_catalog_example_dispatches_the_exact_recurrence_value() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "holonomic.differential_operator.to_coefficient_recurrence.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.valid_from == 0
    assert [term.shift for term in result.recurrence] == [0, 1]
