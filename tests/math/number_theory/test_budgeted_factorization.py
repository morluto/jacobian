from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._factorization_kernels import factorize_with_budget
from jacobian.math.number_theory._models import (
    BudgetedFactorizationRequest,
    BudgetedFactorizationResult,
    CertifiedFactorComponent,
)


def test_semiprime_and_bounded_perfect_power_factor_completely() -> None:
    for value in ("10403", str(2**37)):
        result = factorize_with_budget(
            BudgetedFactorizationRequest(value=value, factor_limit=1000)
        )
        assert result.status == "COMPLETE"
        assert math.prod(
            parse_canonical_integer(item.value) ** item.exponent
            for item in result.factors
        ) == int(value)
        assert all(item.status == "CERTIFIED_PRIME" for item in result.factors)


def test_budget_exhaustion_preserves_unfactored_composite_cofactor() -> None:
    value = 1_000_003 * 1_000_033
    result = factorize_with_budget(
        BudgetedFactorizationRequest(value=str(value), factor_limit=4)
    )

    assert result.status == "INCOMPLETE"
    assert result.factors == (result.factors[0],)
    assert result.factors[0].value == str(value)
    assert result.factors[0].status == "UNFACTORED_COMPOSITE"


def test_digit_and_budget_bounds_reject_before_factoring() -> None:
    with pytest.raises(ValidationError):
        BudgetedFactorizationRequest(value="1" + "0" * 15, factor_limit=100)
    with pytest.raises(ValidationError):
        BudgetedFactorizationRequest(value="12", factor_limit=3)


def test_wire_result_binds_every_component_primality_claim() -> None:
    with pytest.raises(
        ValidationError, match="CERTIFIED_PRIME components must be prime"
    ):
        BudgetedFactorizationResult(
            status="COMPLETE",
            value="15",
            factor_limit=4,
            factors=(
                CertifiedFactorComponent(
                    value="15", exponent=1, status="CERTIFIED_PRIME"
                ),
            ),
        )
    with pytest.raises(
        ValidationError, match="UNFACTORED_COMPOSITE components must be composite"
    ):
        BudgetedFactorizationResult(
            status="INCOMPLETE",
            value="13",
            factor_limit=4,
            factors=(
                CertifiedFactorComponent(
                    value="13", exponent=1, status="UNFACTORED_COMPOSITE"
                ),
            ),
        )
