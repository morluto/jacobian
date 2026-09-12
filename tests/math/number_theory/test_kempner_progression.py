"""Independent checks for Kempner arithmetic-progression decisions."""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory._kempner_models import (
    KempnerArithmeticProgressionRequest,
    KempnerArithmeticProgressionResult,
    KempnerDigitSet,
)
from jacobian.math.number_theory._kempner_progression import (
    compute_kempner_arithmetic_progression,
    decide_kempner_arithmetic_progression,
)


def _member(value: int, base: int, digits: tuple[int, ...]) -> bool:
    if value < 1:
        return False
    allowed = set(digits)
    while value:
        value, digit = divmod(value, base)
        if digit not in allowed:
            return False
    return True


def _finite_witness(
    base: int, digits: tuple[int, ...], arity: int, upper: int
) -> tuple[int, int, tuple[int, ...]] | None:
    family = [n for n in range(1, upper + 1) if _member(n, base, digits)]
    family_set = set(family)
    for first, second in combinations(family, 2):
        difference = second - first
        terms = tuple(first + index * difference for index in range(arity))
        if all(term in family_set for term in terms):
            return first, difference, terms
    return None


def test_shortest_deterministic_witness() -> None:
    digit_set = KempnerDigitSet(base=3, allowed_digits=(1, 2))
    result = decide_kempner_arithmetic_progression(digit_set, 3)

    assert result.status == "CONTAINS_PROGRESSION"
    assert result.first_term == 1
    assert result.common_difference == 3
    assert result.indices == (0, 1, 2)
    assert result.values == (1, 4, 7)


def test_restrictive_zero_family_is_progression_free() -> None:
    result = decide_kempner_arithmetic_progression(
        KempnerDigitSet(base=10, allowed_digits=(0,)), 3
    )
    assert result.status == "PROGRESSION_FREE"
    assert result.values == ()


@pytest.mark.parametrize("base", [2, 3, 4, 5])
def test_small_families_agree_with_independent_finite_oracle(base: int) -> None:
    for digits in ((0,), (1,), (0, 1)):
        if any(digit >= base for digit in digits) or len(digits) == base:
            continue
        digit_set = KempnerDigitSet(base=base, allowed_digits=digits)
        result = decide_kempner_arithmetic_progression(digit_set, 3)
        oracle = _finite_witness(base, digits, 3, base**4)
        if oracle is not None:
            assert result.status == "CONTAINS_PROGRESSION"
            assert result.first_term + 2 * result.common_difference == result.values[-1]
            assert all(_member(term, base, digits) for term in result.values)


def test_carry_boundary_witness_replays() -> None:
    with pytest.raises(ValidationError):
        # Full alphabets are intentionally not Kempner-restricted families.
        KempnerDigitSet(base=2, allowed_digits=(0, 1))

    result = decide_kempner_arithmetic_progression(
        KempnerDigitSet(base=4, allowed_digits=(0, 1, 2)), 3
    )
    assert result.status == "CONTAINS_PROGRESSION"
    assert result.first_term is not None and result.common_difference is not None
    assert result.values == tuple(
        result.first_term + index * result.common_difference for index in range(3)
    )
    assert all(_member(term, 4, (0, 1, 2)) for term in result.values)


def test_native_and_json_composition() -> None:
    request = KempnerArithmeticProgressionRequest(
        digit_set=KempnerDigitSet(base=3, allowed_digits=(1, 2)), arity=3
    )
    result = compute_kempner_arithmetic_progression(request)
    round_trip = KempnerArithmeticProgressionResult.model_validate_json(
        result.model_dump_json()
    )
    assert round_trip == result


def test_schema_rejects_noncanonical_digit_sets() -> None:
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": 10, "allowed_digits": [1, 1]})
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": 2, "allowed_digits": [2]})


def test_derived_graph_admission_rejects_before_search() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        decide_kempner_arithmetic_progression(
            KempnerDigitSet(base=64, allowed_digits=(0,)), 4
        )


def test_base_two_arity_four_is_admitted() -> None:
    result = decide_kempner_arithmetic_progression(
        KempnerDigitSet(base=2, allowed_digits=(1,)), 4
    )
    assert result.status in {"PROGRESSION_FREE", "CONTAINS_PROGRESSION"}


def test_oversized_arity_is_a_typed_resource_error() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        decide_kempner_arithmetic_progression(
            KempnerDigitSet(base=2, allowed_digits=(1,)), 10**5000
        )


def test_wire_schema_rejects_base_outside_two_through_sixty_four() -> None:
    schema = KempnerDigitSet.model_json_schema()
    assert schema["properties"]["base"]["pattern"] == r"^(?:[2-9]|[1-5][0-9]|6[0-4])$"
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": "999", "allowed_digits": ["1"]})


def test_progression_free_result_rejects_undefined_arity() -> None:
    with pytest.raises(ValidationError):
        KempnerArithmeticProgressionResult.model_validate(
            {
                "digit_set": {"base": "3", "allowed_digits": ["1"]},
                "arity": "0",
                "conclusion": {"status": "PROGRESSION_FREE"},
            }
        )


def test_result_schema_discriminates_status_branches() -> None:
    schema = KempnerArithmeticProgressionResult.model_json_schema()
    conclusion = schema["properties"]["conclusion"]
    assert "discriminator" in conclusion or "oneOf" in conclusion or "anyOf" in conclusion
    with pytest.raises(ValidationError):
        KempnerArithmeticProgressionResult.model_validate(
            {
                "digit_set": {"base": "10", "allowed_digits": ["0"]},
                "arity": "3",
                "conclusion": {
                    "status": "PROGRESSION_FREE",
                    "values": ["1", "2", "3"],
                },
            }
        )
