"""Regression coverage for the advertised rational recurrence domain."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.sequences.recurrence_solving._models import (
    ClosedFormRequest,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)
from jacobian.math.number_theory.sequences.recurrence_solving._tools import (
    compute_closed_form,
    compute_find_recurrence,
)
from jacobian.math.number_theory.sequences.recurrence_solving.operations import (
    verify_closed_form,
    verify_recurrence,
)


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational(num=numerator, den=denominator)


def test_find_recurrence_accepts_exact_rational_sequence() -> None:
    result = compute_find_recurrence(
        RecurrenceFindRequest(sequence=(_q(1), _q(1, 2), _q(1, 4), _q(1, 8)))
    )
    assert result.status == "FOUND"
    assert result.order == 1
    assert result.coefficients[0].as_integer_ratio() == (1, 2)
    assert result.sequence == (_q(1), _q(1, 2), _q(1, 4), _q(1, 8))
    assert verify_recurrence(result)

    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded.sequence == result.sequence
    assert verify_recurrence(decoded)

    forged = result.model_copy(update={"coefficients": (_q(3),), "order": 1})
    assert not verify_recurrence(forged)


def test_find_recurrence_accepts_no_fitting_when_no_nonvacuous_order_exists() -> None:
    result = compute_find_recurrence(RecurrenceFindRequest(sequence=(_q(0), _q(1))))
    assert result.status == "NO_FITTING_RECURRENCE"
    assert result.order == 0
    assert result.coefficients == ()
    RecurrenceFindResult.model_validate(result.model_dump())


def test_closed_form_accepts_exact_rational_data() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=(_q(1), _q(-1, 2)),
            initial_values=(_q(3, 2),),
        )
    )
    assert result.expression.value == "3*2**(-n - 1)"
    assert result.characteristic_coefficients == (_q(1), _q(-1, 2))
    assert result.initial_values == (_q(3, 2),)
    assert verify_closed_form(result)

    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_closed_form(decoded)

    forged = result.model_copy(
        update={"expression": result.expression.model_copy(update={"value": "0"})}
    )
    assert not verify_closed_form(forged)


def test_closed_form_expression_uses_non_evaluating_wire_grammar() -> None:
    from pydantic import ValidationError

    from jacobian.math.number_theory.sequences.recurrence_solving._models import (
        ClosedFormExpression,
    )

    with pytest.raises(ValidationError):
        ClosedFormExpression(value="__import__('os').system('id')")
