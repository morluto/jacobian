"""Regression coverage for the advertised rational recurrence domain."""

from jacobian.math.recurrence_solving._models import (
    ClosedFormRequest,
    RecurrenceFindRequest,
)
from jacobian.math.recurrence_solving._operations import (
    compute_closed_form,
    compute_find_recurrence,
)


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def test_find_recurrence_accepts_exact_rational_sequence() -> None:
    result = compute_find_recurrence(
        RecurrenceFindRequest(sequence=(_q(1), _q(1, 2), _q(1, 4), _q(1, 8)))
    )
    assert result.status == "FOUND"
    assert result.order == 1
    assert result.coefficients[0].as_integer_ratio() == (1, 2)


def test_closed_form_accepts_exact_rational_data() -> None:
    result = compute_closed_form(
        ClosedFormRequest(
            characteristic_coefficients=(_q(1), _q(-1, 2)),
            initial_values=(_q(3, 2),),
        )
    )
    assert result.expression == "3*2**(-n - 1)"
