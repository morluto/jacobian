"""Regression tests for canonical rational continued fractions (#2312)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic._rational_models import (
    MAX_RATIONAL_CONTINUED_FRACTION_TERMS,
    RationalContinuedFractionResult,
    RationalValueRequest,
)
from jacobian.math.number_theory.arithmetic._rationals import continued_fraction


def _request(num: str, den: str) -> RationalValueRequest:
    return RationalValueRequest(value=CanonicalRational(num=num, den=den))


@pytest.mark.parametrize(
    ("num", "den", "expected"),
    (
        ("7", "3", ("2", "3")),
        ("-7", "3", ("-3", "1", "2")),
        ("5", "1", ("5",)),
        ("0", "1", ("0",)),
        ("1", "2", ("0", "2")),
        ("-1", "2", ("-1", "2")),
        ("3", "1", ("3",)),
    ),
)
def test_producer_emits_canonical_expansions(
    num: str, den: str, expected: tuple[str, ...]
) -> None:
    result = continued_fraction(_request(num, den))
    assert result.terms == expected
    assert result.value == CanonicalRational(num=num, den=den)
    assert RationalContinuedFractionResult.model_validate(result.model_dump()) == result


def test_two_representation_boundary_is_canonical() -> None:
    """The expansion never ends in 1; the merged form is the accepted one."""

    result = continued_fraction(_request("7", "3"))
    assert result.terms[-1] != "1"
    with pytest.raises(ValidationError) as exc_info:
        RationalContinuedFractionResult(
            value=CanonicalRational(num="7", den="3"),
            terms=("2", "2", "1"),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic.continued_fraction_trailing_one"
    )


def test_result_rejects_nonpositive_tail_term() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RationalContinuedFractionResult(
            value=CanonicalRational(num="7", den="3"),
            terms=("0", "0"),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic.continued_fraction_nonpositive_term"
    )


def test_result_rejects_terms_for_a_different_rational() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RationalContinuedFractionResult(
            value=CanonicalRational(num="7", den="3"),
            terms=("2", "2"),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic.continued_fraction_reconstruction"
    )


def test_producer_rejects_expansion_beyond_result_term_bound() -> None:
    previous, current = 0, 1
    for _ in range(MAX_RATIONAL_CONTINUED_FRACTION_TERMS + 2):
        previous, current = current, previous + current

    with pytest.raises(OperationDomainValidationError) as exc_info:
        continued_fraction(_request(str(current), str(previous)))

    assert exc_info.value.errors() == (
        {
            "loc": ("value",),
            "type": "arithmetic.continued_fraction_terms_exceed_limit",
            "msg": (
                "continued fraction exceeds the "
                f"{MAX_RATIONAL_CONTINUED_FRACTION_TERMS}-term result bound"
            ),
        },
    )
