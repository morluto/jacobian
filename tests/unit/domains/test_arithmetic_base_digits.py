"""Focused contract checks for positional integer expansion."""

import pytest
from pydantic import ValidationError

from jacobian.contracts.arithmetic import (
    IntegerBaseDigitsRequest,
    IntegerBaseDigitsResult,
)
from jacobian.domains.arithmetic import operations as arithmetic_operations
from jacobian.domains.arithmetic.operations import base_digits


@pytest.mark.parametrize(
    ("value", "base", "expected"),
    (
        ("10", 2, IntegerBaseDigitsResult(sign=1, base=2, digits=("1", "0", "1", "0"))),
        (
            "-10",
            2,
            IntegerBaseDigitsResult(sign=-1, base=2, digits=("1", "0", "1", "0")),
        ),
        ("0", 2, IntegerBaseDigitsResult(sign=0, base=2, digits=("0",))),
        (
            "9999",
            10_000,
            IntegerBaseDigitsResult(sign=1, base=10_000, digits=("9999",)),
        ),
    ),
)
def test_base_digits_separates_sign_base_and_digits(
    value: str,
    base: int,
    expected: IntegerBaseDigitsResult,
) -> None:
    result = base_digits(IntegerBaseDigitsRequest(value=value, base=base))

    assert result == expected


@pytest.mark.parametrize(
    "invalid",
    (
        {"sign": 0, "base": 10, "digits": ("1",)},
        {"sign": 1, "base": 10, "digits": ("0",)},
        {"sign": 1, "base": 2, "digits": ("2",)},
    ),
)
def test_base_digits_result_rejects_noncanonical_separated_fields(
    invalid: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        IntegerBaseDigitsResult.model_validate(invalid)


def test_base_digits_rejects_an_oversized_result_before_integer_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_conversion(_: str) -> int:
        raise AssertionError("base expansion must preflight before integer conversion")

    monkeypatch.setattr(arithmetic_operations, "_int", unexpected_conversion)

    with pytest.raises(ValueError, match="1024-digit result bound"):
        base_digits(IntegerBaseDigitsRequest(value="1" + ("0" * 1_024), base=10))
