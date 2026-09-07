from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics import _difference_set_models as sidon_models
from jacobian.math.combinatorics import _difference_sets as sidon_kernel
from jacobian.math.combinatorics._difference_set_models import (
    MAX_SIDON_ORDERED_DIFFERENCES,
    MAX_SIDON_SET_SIZE,
    CyclicDifferenceMultiplicity,
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    CyclicPerfectDifferenceSetRequest,
    CyclicPerfectDifferenceSetResult,
    IntegerSidonRequest,
    IntegerSidonResult,
)
from jacobian.math.combinatorics._difference_sets import (
    _require_integer_sidon_result_admission,
    decide_cyclic_difference_set_extension,
    decide_cyclic_perfect_difference_set,
    decide_integer_sidon,
)


@contextmanager
def raises_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as exc_info:
        yield
    assert exc_info.value.errors()[0]["type"] == code


def test_sidon_request_rejects_duplicate_integer_elements() -> None:
    with raises_code("combinatorics.sidon_invariant"):
        IntegerSidonRequest(elements=(1, 2, 1))


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_integer_set_schemas_retain_canonical_signed_bounds(mode: Any) -> None:
    for model, field in (
        (IntegerSidonRequest, "elements"),
        (IntegerSidonResult, "normalized_elements"),
        (CyclicDifferenceSetExtensionRequest, "base_elements"),
    ):
        schema = model.model_json_schema(mode=mode)["properties"][field]["items"]
        validator = Draft202012Validator(schema)
        for accepted in ("0", "9" * 128, "-" + "9" * 127):
            assert validator.is_valid(accepted)
        for rejected in (0, "01", "-0", "1\n", "9" * 129, "-" + "9" * 128):
            assert not validator.is_valid(rejected)


@pytest.mark.parametrize("sign", [1, -1])
def test_integer_set_signed_bound_matches_native_and_json(sign: int) -> None:
    digits = 128 - int(sign < 0)
    accepted = sign * (10**digits - 1)
    rejected = sign * 10**digits
    for model, field, extra in (
        (IntegerSidonRequest, "elements", {}),
        (CyclicDifferenceSetExtensionRequest, "base_elements", {"target_order": 3}),
    ):
        native = model.model_validate({field: (accepted,), **extra})
        assert model.model_validate_json(native.model_dump_json()) == native
        with pytest.raises(ValidationError):
            model.model_validate({field: (rejected,), **extra})
        with pytest.raises(ValidationError):
            model.model_validate_json(json.dumps({field: [str(rejected)], **extra}))


def test_sidon_result_keeps_structural_normalization() -> None:
    produced = decide_integer_sidon(IntegerSidonRequest(elements=(4, 1, 2)))
    result = IntegerSidonResult.model_validate_json(produced.model_dump_json())
    assert result.normalized_elements == (1, 2, 4)


def test_sidon_kernel_returns_the_complete_ordered_difference_profile() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=(0, 1, 3)))
    expected = tuple(
        (left, right, left - right)
        for left in (0, 1, 3)
        for right in (0, 1, 3)
        if left != right
    )
    assert (
        tuple(
            (int(item.minuend), int(item.subtrahend), int(item.difference))
            for item in result.ordered_differences
        )
        == expected
    )


def test_sidon_kernel_decision_matches_its_complete_profile() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=(0, 1, 3)))
    differences = tuple(int(item.difference) for item in result.ordered_differences)
    assert result.is_sidon == (len(set(differences)) == len(differences))


def test_singleton_sidon_has_empty_complete_ledger() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=(7,)))

    assert result.normalized_elements == (7,)
    assert result.ordered_differences == ()
    assert result.is_sidon is True


def test_sidon_admits_complete_profile_for_first_69_squares() -> None:
    elements = tuple(value * value for value in range(1, 70))
    result = decide_integer_sidon(IntegerSidonRequest(elements=elements))

    assert len(result.normalized_elements) == 69
    assert len(result.ordered_differences) == 69 * 68


def test_sidon_parser_ceiling_is_a_direct_cardinality_bound() -> None:
    admitted = tuple(range(MAX_SIDON_SET_SIZE))
    rejected = tuple(value for value in range(MAX_SIDON_SET_SIZE + 1))

    assert MAX_SIDON_ORDERED_DIFFERENCES == MAX_SIDON_SET_SIZE * (
        MAX_SIDON_SET_SIZE - 1
    )
    IntegerSidonRequest(elements=admitted)
    _require_integer_sidon_result_admission(admitted)
    with pytest.raises(ValidationError) as exc_info:
        IntegerSidonRequest(elements=rejected)
    assert exc_info.value.errors()[0]["type"] == "too_long"


def test_sidon_translated_interval_reaches_result_sensitive_admission() -> None:
    elements = tuple(range(-9, 426))
    request = IntegerSidonRequest(elements=elements)
    plan = _require_integer_sidon_result_admission(elements)

    assert len(request.elements) == 435
    assert MAX_SIDON_SET_SIZE == 435
    assert len(plan.differences) == MAX_SIDON_ORDERED_DIFFERENCES


def test_sidon_nonnegative_interval_at_the_ceiling_is_admitted() -> None:
    elements = tuple(range(MAX_SIDON_SET_SIZE))
    IntegerSidonRequest(elements=elements)

    plan = _require_integer_sidon_result_admission(elements)
    assert len(plan.differences) == MAX_SIDON_ORDERED_DIFFERENCES


def test_sidon_zero_through_256_reaches_result_sensitive_admission() -> None:
    elements = tuple(range(257))
    request = IntegerSidonRequest(elements=elements)

    assert len(request.elements) == 257
    _require_integer_sidon_result_admission(elements)


def test_sidon_kernel_reuses_admitted_difference_wires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    captured: list[Any] = []
    real_profile = sidon_models._integer_sidon_profile

    def counted_profile(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        plan = real_profile(*args, **kwargs)
        captured.append(plan)
        return plan

    monkeypatch.setattr(sidon_models, "_integer_sidon_profile", counted_profile)
    monkeypatch.setattr(sidon_kernel, "_integer_sidon_profile", counted_profile)
    result = decide_integer_sidon(IntegerSidonRequest(elements=(0, 1, 3)))

    assert calls == 1
    plan = captured[0]
    assert result.normalized_elements == plan.normalized_elements
    assert result.is_sidon == plan.is_sidon
    assert (
        tuple(
            (item.minuend, item.subtrahend, item.difference)
            for item in result.ordered_differences
        )
        == plan.differences
    )


def test_sidon_admits_wide_elements_within_the_cardinality_bound() -> None:
    prefix = "9" * 125
    elements = tuple(int(f"{prefix}{value:03d}") for value in range(256))

    result = decide_integer_sidon(
        IntegerSidonRequest(elements=tuple(int(value) for value in elements))
    )
    assert len(result.ordered_differences) == 256 * 255


def test_sidon_result_rejects_forged_difference_and_decision() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=(0, 1, 3)))
    payload = result.model_dump(mode="json")
    payload["ordered_differences"][0]["difference"] = "0"
    payload["is_sidon"] = not result.is_sidon

    with raises_code("combinatorics.sidon_invariant"):
        IntegerSidonResult.model_validate_json(json.dumps(payload))


def test_extension_request_rejects_an_unbounded_candidate_space() -> None:
    request = CyclicDifferenceSetExtensionRequest(
        base_elements=(0, 1, 2, 3, 4, 5, 6),
        target_order=10,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        decide_cyclic_difference_set_extension(request)
    assert error.value.errors() == (
        {
            "loc": ("base_elements", "target_order"),
            "type": "combinatorics.extension_candidate_space_bound",
            "msg": "extension candidate space exceeds the complete-search bound",
        },
    )


def test_pds_result_accepts_the_canonical_fano_profile() -> None:
    residues = (0, 1, 3)
    modulus = 7
    counts = Counter(
        (left - right) % modulus
        for left in residues
        for right in residues
        if left != right
    )
    profile = tuple(
        CyclicDifferenceMultiplicity(
            residue=residue, multiplicity=counts.get(residue, 0)
        )
        for residue in range(1, modulus)
    )
    missing = tuple(
        residue for residue in range(1, modulus) if counts.get(residue, 0) == 0
    )
    repeated = tuple(
        residue for residue in range(1, modulus) if counts.get(residue, 0) > 1
    )
    result = CyclicPerfectDifferenceSetResult(
        modulus=modulus,
        normalized_residues=residues,
        order=len(residues),
        expected_modulus=modulus,
        difference_multiplicities=profile,
        missing_residues=missing,
        repeated_residues=repeated,
        is_perfect=True,
    )
    assert result.is_perfect is True


def test_pds_kernel_returns_the_complete_profile_for_its_source() -> None:
    result = decide_cyclic_perfect_difference_set(
        CyclicPerfectDifferenceSetRequest(modulus=7, residues=(0, 1, 3))
    )
    counts = Counter(
        (left - right) % result.modulus
        for left in result.normalized_residues
        for right in result.normalized_residues
        if left != right
    )
    assert tuple(
        item.multiplicity for item in result.difference_multiplicities
    ) == tuple(counts.get(residue, 0) for residue in range(1, result.modulus))


def test_extension_result_rejects_a_witness_that_drops_the_retained_base() -> None:
    result = decide_cyclic_difference_set_extension(
        CyclicDifferenceSetExtensionRequest(base_elements=(0, 1), target_order=3)
    )
    assert result.decision == "EXTENDS"
    payload = result.model_dump(mode="json")
    payload["extension"] = [0, 2, 4]
    with pytest.raises(ValidationError):
        CyclicDifferenceSetExtensionResult.model_validate_json(json.dumps(payload))
