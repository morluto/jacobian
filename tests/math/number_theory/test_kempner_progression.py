"""Independent checks for Kempner arithmetic-progression decisions."""

from __future__ import annotations

from itertools import combinations

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import (
    _KEMPNER_ARITY_PATTERN,
    _KEMPNER_BASE_PATTERN,
    MAX_KEMPNER_ARITY,
    MIN_KEMPNER_ARITY,
    KempnerArithmeticProgressionRequest,
    KempnerArithmeticProgressionResult,
    KempnerContainsProgression,
    KempnerDigitSet,
    KempnerProgressionFree,
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
            KempnerDigitSet(base=64, allowed_digits=(1, 63)), 12
        )


def test_zero_only_family_is_progression_free_at_every_arity() -> None:
    """The positive family is empty, so no carry graph is needed."""
    for arity in (3, 4, 5, 10):
        result = decide_kempner_arithmetic_progression(
            KempnerDigitSet(base=64, allowed_digits=(0,)), arity
        )
        assert result.conclusion.status == "PROGRESSION_FREE"


def test_singleton_digit_family_is_progression_free_without_search() -> None:
    """Repdigits admit no three-term progression, so no carry graph is needed."""
    for arity in (3, 4, 5, 10):
        result = decide_kempner_arithmetic_progression(
            KempnerDigitSet(base=64, allowed_digits=(1,)), arity
        )
        assert result.conclusion.status == "PROGRESSION_FREE"


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
    assert schema["properties"]["base"]["pattern"] == _KEMPNER_BASE_PATTERN
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": "999", "allowed_digits": ["1"]})
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": "2\n", "allowed_digits": ["1"]})
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors({"base": "2\n", "allowed_digits": ["1"]}))


def test_wire_schema_encodes_arity_lower_bound() -> None:
    request_schema = KempnerArithmeticProgressionRequest.model_json_schema()
    result_schema = KempnerArithmeticProgressionResult.model_json_schema()
    assert request_schema["properties"]["arity"]["pattern"] == _KEMPNER_ARITY_PATTERN
    assert result_schema["properties"]["arity"]["pattern"] == _KEMPNER_ARITY_PATTERN
    validator = Draft202012Validator(request_schema)
    for arity in ("0", "1", "2"):
        payload = {
            "digit_set": {"base": "3", "allowed_digits": ["1"]},
            "arity": arity,
        }
        with pytest.raises(ValidationError):
            KempnerArithmeticProgressionRequest.model_validate(payload)
        assert list(validator.iter_errors(payload))


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
    assert (
        "discriminator" in conclusion or "oneOf" in conclusion or "anyOf" in conclusion
    )
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


def test_wire_schema_bounds_item_specific_witness_values() -> None:
    """Items must advertise the same domain as runtime canonicalization."""
    digit_schema = KempnerDigitSet.model_json_schema()
    digit_items = digit_schema["properties"]["allowed_digits"]["items"]
    assert digit_items["pattern"].startswith("^(?:[0-9]|[1-5][0-9]|6[0-3])")
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": "10", "allowed_digits": ["-1"]})
    with pytest.raises(ValidationError):
        KempnerDigitSet.model_validate({"base": "10", "allowed_digits": ["999"]})

    witness_schema = KempnerContainsProgression.model_json_schema()
    index_items = witness_schema["properties"]["indices"]["items"]
    assert index_items["pattern"].startswith(
        "^(?:[0-9]|[1-9][0-9]|[1-8][0-9]{2}|9[0-8][0-9]|99[0-8])"
    )
    with pytest.raises(ValidationError):
        KempnerContainsProgression.model_validate(
            {
                "status": "CONTAINS_PROGRESSION",
                "indices": ["999", "1", "2"],
                "values": ["1", "2", "3"],
                "first_term": "1",
                "common_difference": "1",
            }
        )
    with pytest.raises(ValidationError):
        KempnerContainsProgression.model_validate(
            {
                "status": "CONTAINS_PROGRESSION",
                "indices": ["0", "1", "2"],
                "values": ["0", "1", "2"],
                "first_term": "1",
                "common_difference": "1",
            }
        )
    for field in ("first_term", "common_difference"):
        assert witness_schema["properties"][field]["ge"] == 1
    base_witness = {
        "status": "CONTAINS_PROGRESSION",
        "indices": ["0", "1", "2"],
        "values": ["1", "2", "3"],
        "first_term": "1",
        "common_difference": "1",
    }
    for field, bad in (
        ("indices", ["-1", "1", "2"]),
        ("first_term", "0"),
        ("common_difference", "-1"),
    ):
        payload = dict(base_witness)
        payload[field] = bad
        with pytest.raises(ValidationError):
            KempnerContainsProgression.model_validate(payload)


def test_witness_arrays_are_capped_at_max_arity() -> None:
    schema = KempnerContainsProgression.model_json_schema()
    assert schema["properties"]["indices"]["maxItems"] == MAX_KEMPNER_ARITY
    assert schema["properties"]["values"]["maxItems"] == MAX_KEMPNER_ARITY
    assert schema["properties"]["indices"]["minItems"] == MIN_KEMPNER_ARITY
    assert schema["properties"]["values"]["minItems"] == MIN_KEMPNER_ARITY
    too_long = MAX_KEMPNER_ARITY + 1
    with pytest.raises(ValidationError) as error:
        KempnerContainsProgression.model_validate(
            {
                "status": "CONTAINS_PROGRESSION",
                "indices": list(range(too_long)),
                "values": list(range(1, too_long + 1)),
                "first_term": 1,
                "common_difference": 1,
            }
        )
    assert {item["type"] for item in error.value.errors()} & {
        "too_long",
        "tuple_too_long",
    }


def test_serialized_branches_require_status_discriminators() -> None:
    assert "status" in KempnerProgressionFree.model_json_schema()["required"]
    assert "status" in KempnerContainsProgression.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        KempnerArithmeticProgressionResult.model_validate(
            {
                "digit_set": {"base": "10", "allowed_digits": ["0"]},
                "arity": "3",
                "conclusion": {},
            }
        )


def test_zero_only_shortcut_still_validates_the_domain() -> None:
    """A constructed out-of-domain digit set is refused before the presolve."""
    from jacobian.catalog.models import OperationDomainValidationError

    forged = KempnerDigitSet.model_construct(base=1, allowed_digits=(0,))
    with pytest.raises(OperationDomainValidationError) as error:
        decide_kempner_arithmetic_progression(forged, 4)
    assert error.value.errors()[0]["type"] == (
        "number_theory.kempner_progression.canonical_digit_set"
    )


def test_zero_only_shortcut_rejects_oversized_arity() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        decide_kempner_arithmetic_progression(
            KempnerDigitSet(base=64, allowed_digits=(0,)), 1_000
        )
    assert error.value.errors()[0]["type"] == (
        "number_theory.kempner_progression.arity_bound"
    )


def test_forged_digit_entries_are_rejected_without_a_type_error() -> None:
    """A constructed digit set with an unhashable entry is a typed error."""
    from jacobian.catalog.models import OperationDomainValidationError

    forged = KempnerDigitSet.model_construct(base=10, allowed_digits=([],))
    with pytest.raises(OperationDomainValidationError) as error:
        decide_kempner_arithmetic_progression(forged, 3)
    assert error.value.errors()[0]["type"] == (
        "number_theory.kempner_progression.canonical_digit_set"
    )


def test_digit_level_progression_is_presolved_before_graph_admission() -> None:
    """Digits 1..6 at arity 6 are a one-digit progression, not a graph search."""
    result = decide_kempner_arithmetic_progression(
        KempnerDigitSet(base=7, allowed_digits=(1, 2, 3, 4, 5, 6)), 6
    )
    assert isinstance(result.conclusion, KempnerContainsProgression)
    assert result.conclusion.values == (1, 2, 3, 4, 5, 6)


def test_source_reachable_witness_bound_admits_two_digit_family() -> None:
    """A two-column witness is admitted without charging the full state count."""
    digit_set = KempnerDigitSet(base=5, allowed_digits=(1, 2, 3))
    result = decide_kempner_arithmetic_progression(digit_set, 4)

    assert result.status == "CONTAINS_PROGRESSION"
    assert result.first_term == 1
    assert result.common_difference == 5
    assert result.values == (1, 6, 11, 16)
    independent = _finite_witness(5, (1, 2, 3), 4, 16)
    assert independent is not None
    assert result.values == independent[2]


def test_oversized_constructed_digit_set_is_rejected_before_scanning() -> None:
    """A forged oversized digit tuple is rejected by length before inspection."""

    class _NoIterTuple(tuple):
        def __iter__(self):
            raise AssertionError("oversized digit set was scanned")

    digit_set = KempnerDigitSet.model_construct(
        base=64, allowed_digits=_NoIterTuple((0,) * 64)
    )
    with pytest.raises(OperationDomainValidationError):
        decide_kempner_arithmetic_progression(digit_set, 3)


def test_multi_column_progression_is_admitted_before_graph_rejection() -> None:
    """A cheap multi-column witness is found before the combinatorial bound."""
    digit_set = KempnerDigitSet(base=4, allowed_digits=(0, 1, 2))
    result = decide_kempner_arithmetic_progression(digit_set, 6)
    assert result.status == "CONTAINS_PROGRESSION"
    values = result.values
    assert len(values) == 6
    assert all(_member(value, 4, (0, 1, 2)) for value in values)
    differences = {values[index + 1] - values[index] for index in range(5)}
    assert len(differences) == 1


def test_expensive_presolve_is_admitted_before_the_carry_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unadmitted presolve must not run before semantic admission rejects.

    Base 64 with all nonzero digits and arity 999 would scan roughly
    ``512 * 64**2 * 999`` transitions. The presolve work bound is checked
    first, so the carry search is never entered and the resource admission
    owns the rejection.
    """
    from jacobian.math.number_theory import _kempner_progression as module

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the unadmitted presolve must not run")

    monkeypatch.setattr(module, "_carry_witness", forbidden)
    digit_set = KempnerDigitSet(base=64, allowed_digits=tuple(range(63)))
    with pytest.raises(OperationResourceAdmissionError):
        decide_kempner_arithmetic_progression(digit_set, 999)
