import pytest
import sympy as sp

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.differential_left_division._models import (
    DifferentialLeftDivisionRequest,
)
from jacobian.math.ore_algebras.differential_left_division._tools import TOOLS
from jacobian.math.ore_algebras.differential_left_division.operations import (
    differential_operator_left_divide_monic,
)
from jacobian.math.ore_algebras.operations import (
    differential_operator_add,
    differential_operator_multiply,
)
from jacobian.math.polynomials._conversions import rational_function_to_sympy


def _coefficient(terms: list[tuple[int, int]]) -> dict:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [
                {
                    "coefficient": {"num": value, "den": 1},
                    "exponents": [degree],
                }
                for degree, value in sorted(terms, reverse=True)
                if value
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }


def _operator(*terms: tuple[int, list[tuple[int, int]]]) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {"order": order, "coefficient": _coefficient(coefficient)}
                for order, coefficient in terms
            ],
        }
    )


def _action(operator: DifferentialOreOperator, function, x):
    return sum(
        rational_function_to_sympy(term.coefficient, symbols=(x,))
        * sp.diff(function, x, term.order)
        for term in operator.terms
    )


def test_left_division_reconstructs_exact_noncommutative_operator() -> None:
    x = sp.Symbol("x")
    dividend = _operator(
        (0, [(1, 1), (0, 1)]),
        (1, [(2, 1), (0, 2)]),
        (2, [(1, 1)]),
    )
    divisor = _operator((0, [(1, 1)]), (1, [(0, 1)]))

    result = differential_operator_left_divide_monic(dividend, divisor)

    assert result.quotient == _operator((0, [(0, 1)]), (1, [(1, 1)]))
    assert result.remainder == _operator((0, [(0, 1)]))
    product = differential_operator_multiply(divisor, result.quotient).product
    reconstructed = differential_operator_add(product, result.remainder).sum
    assert reconstructed == dividend
    for degree in range(9):
        polynomial = x**degree
        assert (
            sp.expand(
                _action(dividend, polynomial, x)
                - _action(divisor, _action(result.quotient, polynomial, x), x)
                - _action(result.remainder, polynomial, x)
            )
            == 0
        )


def test_left_orientation_differs_from_right_factor_order() -> None:
    dividend = _operator(
        (0, [(1, 1), (0, 1)]),
        (1, [(2, 1), (0, 2)]),
        (2, [(1, 1)]),
    )
    divisor = _operator((0, [(1, 1)]), (1, [(0, 1)]))
    result = differential_operator_left_divide_monic(dividend, divisor)

    right_product = differential_operator_multiply(result.quotient, divisor).product
    assert right_product != dividend


def test_zero_lower_order_and_order_zero_divisors_compose() -> None:
    dividend = _operator((0, [(2, 1)]), (1, [(0, 1)]))
    lower_order_divisor = _operator((2, [(0, 1)]))
    one = _operator((0, [(0, 1)]))
    zero = _operator()

    lower = differential_operator_left_divide_monic(dividend, lower_order_divisor)
    assert lower.quotient == zero
    assert lower.remainder == dividend
    by_one = differential_operator_left_divide_monic(dividend, one)
    assert by_one.quotient == dividend
    assert by_one.remainder == zero
    zero_dividend = differential_operator_left_divide_monic(zero, lower_order_divisor)
    assert zero_dividend.quotient == zero_dividend.remainder == zero


def test_divisor_and_coefficient_contracts_are_explicit() -> None:
    dividend = _operator((1, [(0, 1)]))
    with pytest.raises(OperationDomainValidationError):
        differential_operator_left_divide_monic(
            dividend,
            _operator((0, [(1, 1)]), (1, [(0, 2)])),
        )
    rational_coefficient = {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [{"coefficient": {"num": 1, "den": 2}, "exponents": [0]}]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
        },
    }
    with pytest.raises(OperationDomainValidationError):
        differential_operator_left_divide_monic(
            DifferentialOreOperator.model_validate(
                {"terms": [{"order": 1, "coefficient": rational_coefficient}]}
            ),
            _operator((1, [(0, 1)])),
        )
    with pytest.raises(OperationDomainValidationError):
        differential_operator_left_divide_monic(dividend, {"terms": []})


def test_manifest_exposes_typed_left_division() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert isinstance(tool, MathTool)
    assert tool.operation_id == "ore.differential.operator.left_division.compute"
    assert tool.request_type is DifferentialLeftDivisionRequest
    assert tool.result_type.__name__ == "DifferentialLeftDivisionResult"
    result = tool.run(tool.request_type.model_validate(tool.examples[0].input))
    assert result.quotient == _operator((0, [(0, 1)]), (1, [(1, 1)]))
