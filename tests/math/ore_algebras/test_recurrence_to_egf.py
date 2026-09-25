import json
from fractions import Fraction

import pytest
import sympy as sp

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras.recurrence_to_egf._tools import TOOLS
from jacobian.math.ore_algebras.recurrence_to_egf.operations import (
    polynomial_recurrence_to_egf_equation,
)


def _rf(terms: list[tuple[int, int]], variable: str = "n") -> dict:
    return {
        "domain": "QQ",
        "variables": [variable],
        "numerator": {
            "terms": [
                {"coefficient": {"num": value, "den": 1}, "exponents": [degree]}
                for degree, value in sorted(terms, reverse=True)
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }


def _recurrence(*terms: tuple[int, list[tuple[int, int]]]) -> dict:
    return {
        "variable": "n",
        "terms": [
            {"exponent": exponent, "coefficient": _rf(polynomial)}
            for exponent, polynomial in terms
        ],
    }


def _polynomial(coefficient) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in coefficient.numerator.terms
    }


def _as_sympy_polynomial(coefficient, variable):
    return sum(
        sp.Rational(value.numerator, value.denominator) * variable**degree
        for degree, value in _polynomial(coefficient).items()
    )


def _apply_operator(result, function, x):
    return sum(
        _as_sympy_polynomial(term.coefficient, x) * sp.diff(function, x, term.order)
        for term in result.differential_operator.terms
    )


def test_quadratic_euler_power_uses_stirling_second_kind_identity() -> None:
    # n^2 a_n maps to (xD)^2 E = (xD + x^2D^2)E.
    result = polynomial_recurrence_to_egf_equation(_recurrence((0, [(2, 1)])))
    assert [
        (term.order, _polynomial(term.coefficient))
        for term in result.differential_operator.terms
    ] == [
        (1, {1: Fraction(1)}),
        (2, {2: Fraction(1)}),
    ]
    x = sp.Symbol("x")
    egf = x**2 / 2
    assert sp.expand(_apply_operator(result, egf, x)) == 2 * x**2
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_shifted_egf_operator_matches_recurrence_residual() -> None:
    # n*a_(n+1)-a_n has EGF residual x E'' - E.
    result = polynomial_recurrence_to_egf_equation(
        _recurrence((0, [(0, -1)]), (1, [(1, 1)]))
    )
    x = sp.Symbol("x")
    egf = sp.exp(x)
    assert sp.simplify(_apply_operator(result, egf, x) - (x - 1) * egf) == 0
    assert [
        (term.order, _polynomial(term.coefficient))
        for term in result.differential_operator.terms
    ] == [(0, {0: Fraction(-1)}), (2, {1: Fraction(1)})]


def test_fibonacci_egf_equation_has_no_boundary_forcing() -> None:
    result = polynomial_recurrence_to_egf_equation(
        _recurrence((0, [(0, -1)]), (1, [(0, -1)]), (2, [(0, 1)]))
    )
    assert [
        (term.order, _polynomial(term.coefficient))
        for term in result.differential_operator.terms
    ] == [
        (0, {0: Fraction(-1)}),
        (1, {0: Fraction(-1)}),
        (2, {0: Fraction(1)}),
    ]
    x = sp.Symbol("x")
    egf = sp.exp(x)
    # For a_n=1, the recurrence residual is 1-1-1=-1 at every n.
    assert sp.simplify(_apply_operator(result, egf, x) + egf) == 0


def test_catalog_example_dispatches_the_typed_egf_equation() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "holonomic.shift_recurrence.to_egf_differential_equation.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert [term.order for term in result.differential_operator.terms] == [0, 1, 2]


def test_egf_transform_rejects_nonpolynomial_coefficients_and_high_order() -> None:
    rational = _rf([(0, 1)])
    rational["denominator"] = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        polynomial_recurrence_to_egf_equation(
            {"variable": "n", "terms": [{"exponent": 0, "coefficient": rational}]}
        )

    with pytest.raises(OperationResourceAdmissionError, match="above the order bound"):
        polynomial_recurrence_to_egf_equation(_recurrence((0, [(17, 1)])))
