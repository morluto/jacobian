import json
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    PolynomialRecurrencePrefixRequest,
    ShiftOreOperator,
)
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import polynomial_recurrence_generate_prefix


def _rf(terms: list[tuple[int, int]]) -> dict:
    return {
        "domain": "QQ",
        "variables": ["n"],
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


def _op(terms: list[tuple[int, list[tuple[int, int]]]]) -> ShiftOreOperator:
    return ShiftOreOperator.model_validate(
        {
            "terms": [
                {"exponent": exponent, "coefficient": _rf(coefficient)}
                for exponent, coefficient in terms
            ]
        }
    )


def test_fibonacci_prefix_obeys_finite_declared_recurrence() -> None:
    result = polynomial_recurrence_generate_prefix(
        _op([(0, [(-1, 0)]), (1, [(-1, 0)]), (2, [(1, 0)])]),
        0,
        FiniteRationalSequence.model_validate({"values": [0, 1]}),
        5,
    )
    values = [item.as_fraction() for item in result.values.values]
    assert values == [0, 1, 1, 2, 3, 5, 8]
    assert result.recurrence_indices == (0, 1, 2, 3, 4)
    assert all(
        values[n + 2] - values[n + 1] - values[n] == 0
        for n in result.recurrence_indices
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_alternating_recurrence_preserves_rational_values() -> None:
    result = polynomial_recurrence_generate_prefix(
        _op([(0, [(1, 0)]), (1, [(1, 0)])]),
        -2,
        FiniteRationalSequence.model_validate({"values": [{"num": 2, "den": 3}]}),
        4,
    )
    assert [value.as_fraction() for value in result.values.values] == [
        Fraction(2, 3),
        Fraction(-2, 3),
        Fraction(2, 3),
        Fraction(-2, 3),
        Fraction(2, 3),
    ]
    assert result.recurrence_indices == (-2, -1, 0, 1)


def test_singular_leading_coefficient_rejected_for_whole_finite_interval() -> None:
    # (n - 1) S + 1 is solvable at n=0 but singular at the second requested index.
    with pytest.raises(
        OperationDomainValidationError, match="leading recurrence coefficient vanishes"
    ):
        polynomial_recurrence_generate_prefix(
            _op([(0, [(1, 0)]), (1, [(1, 1), (-1, 0)])]),
            0,
            FiniteRationalSequence.model_validate({"values": [1]}),
            3,
        )


def test_catalog_example_executes_as_finite_recurrence_value() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "ore.shift.recurrence.generate_finite_prefix.compute"
    )
    request = PolynomialRecurrencePrefixRequest.model_validate_json(
        json.dumps(operation.examples[0].input)
    )
    result = operation.run(request)
    assert [value.as_fraction() for value in result.values.values] == [
        0,
        1,
        1,
        2,
        3,
        5,
        8,
    ]
