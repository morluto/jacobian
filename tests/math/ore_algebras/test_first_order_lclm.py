import json

import pytest
import sympy as sp

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.first_order_lclm._tools import TOOLS
from jacobian.math.ore_algebras.first_order_lclm.operations import (
    differential_first_order_lclm,
)
from jacobian.math.ore_algebras.operations import differential_operator_multiply
from jacobian.math.polynomials._conversions import rational_function_to_sympy


def _coefficient(terms: list[tuple[int, int]], *, denominator=None) -> dict:
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
        "denominator": denominator
        or {"terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]},
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


def _first_order_product(left, right, x):
    """Independent coefficient formula from D*a = a*D + a'."""
    left_coefficients = {
        term.order: rational_function_to_sympy(term.coefficient, symbols=(x,))
        for term in left.terms
    }
    right_coefficients = {
        term.order: rational_function_to_sympy(term.coefficient, symbols=(x,))
        for term in right.terms
    }
    p0, p1 = left_coefficients.get(0, 0), left_coefficients.get(1, 0)
    q0, q1 = right_coefficients.get(0, 0), right_coefficients.get(1, 0)
    return {
        0: sp.cancel(p1 * sp.diff(q0, x) + p0 * q0),
        1: sp.cancel(p1 * sp.diff(q1, x) + p1 * q0 + p0 * q1),
        2: sp.cancel(p1 * q1),
    }


def test_weyl_relation_xd_equals_dx_minus_one() -> None:
    x = sp.Symbol("x")
    f = sp.Function("f")(x)
    multiplication_by_x = _operator((0, [(1, 1)]))
    derivative = _operator((1, [(0, 1)]))

    dx = differential_operator_multiply(derivative, multiplication_by_x).product
    xd = differential_operator_multiply(multiplication_by_x, derivative).product

    assert [term.order for term in dx.terms] == [0, 1]
    assert [term.order for term in xd.terms] == [1]
    assert sp.simplify(_action(xd, f, x) - _action(dx, f, x) + f) == 0


def test_first_order_lclm_witnesses_both_products_and_is_minimal() -> None:
    left = _operator((0, [(1, -1)]), (1, [(0, 1)]))
    right = _operator((0, [(1, 1)]), (1, [(0, 1)]))

    result = differential_first_order_lclm(left, right)
    left_product = differential_operator_multiply(
        result.left_multiplier, result.left
    ).product
    right_product = differential_operator_multiply(
        result.right_multiplier, result.right
    ).product

    assert left_product == right_product == result.common_left_multiple
    x = sp.Symbol("x")
    left_oracle = _first_order_product(result.left_multiplier, result.left, x)
    right_oracle = _first_order_product(result.right_multiplier, result.right, x)
    common_oracle = {
        term.order: rational_function_to_sympy(term.coefficient, symbols=(x,))
        for term in result.common_left_multiple.terms
    }
    assert all(
        sp.cancel(left_oracle.get(order, 0) - right_oracle.get(order, 0)) == 0
        and sp.cancel(left_oracle.get(order, 0) - common_oracle.get(order, 0)) == 0
        for order in range(3)
    )
    assert result.common_left_multiple.order == 2
    assert sp.simplify(_action(result.common_left_multiple, sp.exp(x**2 / 2), x)) == 0
    assert (
        sp.simplify(_action(result.common_left_multiple, sp.exp(-(x**2) / 2), x)) == 0
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_proportional_inputs_have_order_one_lclm() -> None:
    left = _operator((0, [(1, -1)]), (1, [(0, 1)]))
    right = _operator((0, [(1, -2)]), (1, [(0, 2)]))

    result = differential_first_order_lclm(left, right)

    assert result.common_left_multiple == left
    assert result.common_left_multiple.order == 1
    assert (
        differential_operator_multiply(result.left_multiplier, left).product
        == differential_operator_multiply(result.right_multiplier, right).product
        == left
    )


def test_first_order_lclm_rejects_nonpolynomial_coefficients_and_wrong_order() -> None:
    rational_denominator = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    rational = _operator(
        (0, [(0, -1)]),
        (1, [(0, 1)]),
    )
    rational = DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": 0,
                    "coefficient": _coefficient(
                        [(0, -1)], denominator=rational_denominator
                    ),
                },
                rational.terms[1].model_dump(),
            ],
        }
    )
    first_order = _operator((0, [(0, -1)]), (1, [(0, 1)]))
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        differential_first_order_lclm(rational, first_order)

    second_order = _operator((0, [(0, 1)]), (2, [(0, 1)]))
    with pytest.raises(OperationDomainValidationError, match="first-order"):
        differential_first_order_lclm(second_order, second_order)


def test_catalog_example_returns_its_declared_lclm_value() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "ore.differential.operator.first_order_lclm.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.common_left_multiple.order == 2


def test_lclm_product_preflight_does_not_repeat_generic_multiply_admission() -> None:
    left = _operator(
        (0, [(2, -71), (4, -83)]),
        (1, [(1, -97), (3, -61)]),
    )
    right = _operator(
        (0, [(0, -89), (1, -67), (4, 67)]),
        (1, [(1, -89), (3, -83), (4, 73)]),
    )
    # Integer fixture uses the reported coefficient pattern and stays in the
    # declared two-digit input envelope; operation preflight owns admission.
    result = differential_first_order_lclm(left, right)
    assert result.common_left_multiple.order == 2


def test_noncanonical_rational_input_is_a_domain_error() -> None:
    denominator = {
        "terms": [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [0]},
        ]
    }
    shared_factor = _operator((0, [(1, 1)]), (1, [(0, 1)]))
    malformed = DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": 0,
                    "coefficient": _coefficient([(1, 1)], denominator=denominator),
                },
                shared_factor.terms[1].model_dump(),
            ],
        }
    )
    with pytest.raises(OperationDomainValidationError):
        differential_first_order_lclm(malformed, shared_factor)
