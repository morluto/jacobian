import json
from fractions import Fraction
from math import factorial

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    DifferentialOreOperator,
    PolynomialRecurrencePrefixRequest,
)
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import (
    differential_operator_to_coefficient_recurrence,
    polynomial_recurrence_generate_prefix,
)


def _rf(terms: list[tuple[int, int]]) -> dict:
    return {
        "domain": "QQ",
        "variables": ["x"],
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
        (term.exponent, _polynomial(term.coefficient))
        for term in result.recurrence.terms
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
            / factorial(n + term.exponent)
            for term in result.recurrence.terms
        )
        assert lhs == 0
    prefix = polynomial_recurrence_generate_prefix(
        result.recurrence,
        result.valid_from,
        FiniteRationalSequence.model_validate({"values": [1]}),
        5,
    )
    assert [value.as_fraction() for value in prefix.values.values] == [
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 6),
        Fraction(1, 24),
        Fraction(1, 120),
    ]


def test_sinh_equation_has_exact_second_order_recurrence() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (2, [(0, 1)]))
    )
    recurrence = {
        term.exponent: _polynomial(term.coefficient) for term in result.recurrence.terms
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
    assert result.recurrence.terms[0].exponent == 0
    assert _polynomial(result.recurrence.terms[0].coefficient) == {
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
    assert [term.exponent for term in result.recurrence.terms] == [0, 1]


def test_multiple_mixed_boundary_rows_match_direct_coefficient_extraction() -> None:
    # L = 1 + x*D + x^2*D^2 gives a_0=0, 2*a_1=0, then
    # (n^2+1)*a_n=0 for n >= 2.
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, 1)]), (1, [(1, 1)]), (2, [(2, 1)]))
    )
    assert result.valid_from == 2
    assert [
        (
            row.degree,
            [(term.index, term.coefficient.as_fraction()) for term in row.terms],
        )
        for row in result.boundary_rows
    ] == [
        (0, [(0, Fraction(1))]),
        (1, [(1, Fraction(2))]),
    ]
    stable = result.recurrence.terms
    assert len(stable) == 1
    assert stable[0].exponent == 0
    assert _polynomial(stable[0].coefficient) == {0: Fraction(1), 2: Fraction(1)}

    coefficients = [Fraction(2), Fraction(-3), Fraction(5), Fraction(7), Fraction(11)]
    for row in result.boundary_rows:
        lhs = sum(
            term.coefficient.as_fraction() * coefficients[term.index]
            for term in row.terms
        )
        m = row.degree
        direct = (
            coefficients[m]
            + (m * coefficients[m] if m >= 1 else 0)
            + (m * (m - 1) * coefficients[m] if m >= 2 else 0)
        )
        assert lhs == direct
    n = 2
    lhs = sum(
        sum(
            coefficient * n**power
            for power, coefficient in _polynomial(term.coefficient).items()
        )
        * coefficients[n + term.exponent]
        for term in stable
    )
    direct = coefficients[n] + n * coefficients[n] + n * (n - 1) * coefficients[n]
    assert lhs == direct


def test_boundary_work_is_admitted_before_any_boundary_expansion() -> None:
    coefficient = [(degree, 1) for degree in range(1, 65)]
    operator = _operator(*((order, coefficient) for order in range(5)))
    with pytest.raises(OperationResourceAdmissionError, match="work budget"):
        differential_operator_to_coefficient_recurrence(operator)


def test_transform_result_composes_unchanged_with_prefix_generation() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(0, -1)]), (1, [(0, 1)]))
    )
    request = {
        "operator": result.recurrence.model_dump(),
        "start_index": result.valid_from,
        "initial_values": {"values": [1]},
        "steps": 5,
    }
    prefix_request = PolynomialRecurrencePrefixRequest.model_validate(request)
    assert [
        value.as_fraction()
        for value in polynomial_recurrence_generate_prefix(
            prefix_request.operator,
            prefix_request.start_index,
            prefix_request.initial_values,
            prefix_request.steps,
        ).values.values
    ] == [
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 6),
        Fraction(1, 24),
        Fraction(1, 120),
    ]


def test_generated_coefficients_stay_within_downstream_shift_envelope() -> None:
    # Although each input coefficient fits 64 digits, 10^63*(n+16)_16 has
    # a 78-digit constant coefficient and cannot be consumed as a shift op.
    operator = _operator((0, [(0, 1)]), (16, [(0, 10**63)]))
    with pytest.raises(OperationResourceAdmissionError, match="coefficient recurrence coefficient"):
        differential_operator_to_coefficient_recurrence(operator)


def test_output_uses_canonical_shift_operator_through_maximum_shift_span() -> None:
    result = differential_operator_to_coefficient_recurrence(
        _operator((0, [(64, 1)]), (16, [(0, 1)]))
    )
    assert result.recurrence.order == 80
    assert (
        type(result.recurrence).model_validate_json(result.recurrence.model_dump_json())
        == result.recurrence
    )
