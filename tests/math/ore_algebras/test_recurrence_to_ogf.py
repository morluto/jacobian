import json
from fractions import Fraction

import pytest
import sympy as sp

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras.operations import (
    differential_operator_to_coefficient_recurrence,
)
from jacobian.math.ore_algebras.recurrence_to_ogf._tools import TOOLS
from jacobian.math.ore_algebras.recurrence_to_ogf.operations import (
    polynomial_recurrence_to_ogf_equation,
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


def _assert_equation_matches_exact_ogf(result, generating_function) -> None:
    x = sp.Symbol("x")
    left = sum(
        _as_sympy_polynomial(term.coefficient, x)
        * sp.diff(generating_function, x, term.order)
        for term in result.differential_operator.terms
    )
    right = _as_sympy_polynomial(result.forcing, x)
    assert sp.cancel(left - right) == 0


def test_fibonacci_recurrence_yields_ogf_equation_with_boundary_term() -> None:
    result = polynomial_recurrence_to_ogf_equation(
        _recurrence((0, [(0, -1)]), (1, [(0, -1)]), (2, [(0, 1)])),
        {"values": [1, 1]},
    )
    assert [
        (term.order, _polynomial(term.coefficient))
        for term in result.differential_operator.terms
    ] == [(0, {0: Fraction(1), 1: Fraction(-1), 2: Fraction(-1)})]
    assert _polynomial(result.forcing) == {0: Fraction(1)}
    _assert_equation_matches_exact_ogf(
        result, 1 / (1 - sp.Symbol("x") - sp.Symbol("x") ** 2)
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_quadratic_recurrence_uses_correct_euler_operator_expansion() -> None:
    # n^2*a_n maps to (xD)^2 F = xD F + x^2 D^2 F.
    result = polynomial_recurrence_to_ogf_equation(
        _recurrence((0, [(2, 1)])), {"values": []}
    )
    x = sp.Symbol("x")
    generating_function = x**2
    transformed = sum(
        _as_sympy_polynomial(term.coefficient, x)
        * sp.diff(generating_function, x, term.order)
        for term in result.differential_operator.terms
    )
    euler_square = x * sp.diff(x * sp.diff(generating_function, x), x)
    assert sp.expand(transformed - euler_square) == 0
    assert sp.expand(euler_square) == 4 * x**2


def test_polynomial_recurrence_uses_euler_operator_and_keeps_zero_boundary() -> None:
    # (n+1)a_(n+1)-a_n=0 gives xF'-xF=0, with no forcing term.
    result = polynomial_recurrence_to_ogf_equation(
        _recurrence((0, [(0, -1)]), (1, [(0, 1), (1, 1)])),
        {"values": [1]},
    )
    assert [
        (term.order, _polynomial(term.coefficient))
        for term in result.differential_operator.terms
    ] == [(0, {1: Fraction(-1)}), (1, {1: Fraction(1)})]
    assert result.forcing.numerator.terms == ()
    _assert_equation_matches_exact_ogf(result, sp.exp(sp.Symbol("x")))


def test_polynomial_recurrence_keeps_nonzero_initial_boundary_exactly() -> None:
    # n*a_(n+1)-a_n=0 has an initial row -a_0=0. For a_0=1, the
    # transformed inhomogeneous equation records the incompatible boundary.
    result = polynomial_recurrence_to_ogf_equation(
        _recurrence((0, [(0, -1)]), (1, [(1, 1)])), {"values": [1]}
    )
    assert _polynomial(result.forcing) == {0: Fraction(-1)}
    x = sp.Symbol("x")
    arbitrary_f = 1 / (1 - x)
    residual_gf = x / (1 - x) ** 2 - 1 / (1 - x)
    lhs = sum(
        _as_sympy_polynomial(term.coefficient, x) * sp.diff(arbitrary_f, x, term.order)
        for term in result.differential_operator.terms
    )
    assert (
        sp.cancel(lhs - _as_sympy_polynomial(result.forcing, x) - x * residual_gf) == 0
    )


def test_boundary_forcing_growth_is_rejected_before_encoding() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="coefficient bound"):
        polynomial_recurrence_to_ogf_equation(
            _recurrence((1, [(0, 10**63)])),
            {"values": [10**99]},
        )


def test_initial_coefficient_width_is_described_in_request_schema() -> None:
    schema = __import__(
        "jacobian.math.ore_algebras.recurrence_to_ogf._models",
        fromlist=["RecurrenceOGFEquationRequest"],
    ).RecurrenceOGFEquationRequest.model_json_schema()
    field = schema["properties"]["initial_coefficients"]
    assert "a_0 through a_(r-1)" in field["description"]


def test_ogf_differential_equation_round_trips_to_recurrence() -> None:
    transformed = polynomial_recurrence_to_ogf_equation(
        _recurrence((0, [(0, -1)]), (1, [(0, -1)]), (2, [(0, 1)])),
        {"values": [1, 1]},
    )
    reverse = differential_operator_to_coefficient_recurrence(
        transformed.differential_operator
    )
    assert reverse.valid_from == 0
    assert [
        (term.exponent, _polynomial(term.coefficient))
        for term in reverse.recurrence.terms
    ] == [
        (0, {0: Fraction(-1)}),
        (1, {0: Fraction(-1)}),
        (2, {0: Fraction(1)}),
    ]


def test_catalog_example_dispatches_the_typed_equation() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "holonomic.shift_recurrence.to_ogf_differential_equation.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert _polynomial(result.forcing) == {0: Fraction(1)}


def test_ogf_transform_rejects_wrong_initial_width_and_rational_coefficients() -> None:
    with pytest.raises(OperationDomainValidationError, match="initial coefficient"):
        polynomial_recurrence_to_ogf_equation(
            _recurrence((0, [(0, -1)]), (2, [(0, 1)])), {"values": [1]}
        )

    rational = _rf([(0, 1)])
    rational["denominator"] = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        polynomial_recurrence_to_ogf_equation(
            {"variable": "n", "terms": [{"exponent": 0, "coefficient": rational}]},
            {"values": []},
        )


def test_ogf_transform_rejects_unsupported_differential_order_preflight() -> None:
    with pytest.raises(
        OperationResourceAdmissionError, match="differential-operator order"
    ):
        polynomial_recurrence_to_ogf_equation(
            _recurrence((0, [(17, 1)])), {"values": []}
        )
