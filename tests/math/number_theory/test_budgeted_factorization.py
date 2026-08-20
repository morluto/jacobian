from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._factorization_kernels import factorize_with_budget
from jacobian.math.number_theory._models import BudgetedFactorizationRequest


def test_semiprime_and_large_perfect_power_factor_completely() -> None:
    for value in ("10403", str(2**257)):
        result = factorize_with_budget(
            BudgetedFactorizationRequest(value=value, factor_limit=1000)
        )
        assert result.status == "COMPLETE"
        assert math.prod(
            parse_canonical_integer(item.value) ** item.exponent
            for item in result.factors
        ) == int(value)
        assert all(item.status == "CERTIFIED_PRIME" for item in result.factors)


def test_budget_exhaustion_preserves_unresolved_composite_cofactor() -> None:
    value = 1_000_003 * 1_000_033
    result = factorize_with_budget(
        BudgetedFactorizationRequest(value=str(value), factor_limit=4)
    )

    assert result.status == "INCOMPLETE"
    assert result.factors == (result.factors[0],)
    assert result.factors[0].value == str(value)
    assert result.factors[0].status == "UNRESOLVED"


def test_digit_and_budget_bounds_reject_before_factoring() -> None:
    with pytest.raises(ValidationError):
        BudgetedFactorizationRequest(value="1" + "0" * 256, factor_limit=100)
    with pytest.raises(ValidationError):
        BudgetedFactorizationRequest(value="12", factor_limit=3)
