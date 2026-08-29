from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics import _difference_set_models as sidon_models
from jacobian.math.combinatorics import _difference_sets as sidon_kernel
from jacobian.math.combinatorics._difference_set_models import (
    MAX_SIDON_ORDERED_DIFFERENCES,
    MAX_SIDON_RESULT_BYTES,
    MAX_SIDON_SET_SIZE,
    CyclicDifferenceMultiplicity,
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    CyclicPerfectDifferenceSetRequest,
    CyclicPerfectDifferenceSetResult,
    IntegerSidonRequest,
    IntegerSidonResult,
    _integer_sidon_canonical_result_bytes,
    _minimum_integer_sidon_result_bytes,
    _minimum_payload_sidon_elements,
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
        IntegerSidonRequest(elements=("1", "2", "1"))


def test_sidon_result_keeps_structural_normalization() -> None:
    produced = decide_integer_sidon(IntegerSidonRequest(elements=("4", "1", "2")))
    result = IntegerSidonResult.model_validate(produced.model_dump(mode="json"))
    assert result.normalized_elements == ("1", "2", "4")


def test_sidon_kernel_returns_the_complete_ordered_difference_profile() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=("0", "1", "3")))
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
    result = decide_integer_sidon(IntegerSidonRequest(elements=("0", "1", "3")))
    differences = tuple(int(item.difference) for item in result.ordered_differences)
    assert result.is_sidon == (len(set(differences)) == len(differences))


def test_singleton_sidon_has_empty_complete_ledger() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=("7",)))

    assert result.normalized_elements == ("7",)
    assert result.ordered_differences == ()
    assert result.is_sidon is True


def test_sidon_admits_complete_profile_for_first_69_squares() -> None:
    elements = tuple(str(value * value) for value in range(1, 70))
    result = decide_integer_sidon(IntegerSidonRequest(elements=elements))

    assert len(result.normalized_elements) == 69
    assert len(result.ordered_differences) == 69 * 68
    assert len(result.model_dump_json().encode()) < MAX_SIDON_RESULT_BYTES


def test_sidon_byte_formula_matches_compact_json_for_a_false_decision() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=("0", "1", "2")))

    assert result.is_sidon is False
    assert _integer_sidon_canonical_result_bytes((0, 1, 2)) == len(
        result.model_dump_json().encode()
    )


def test_sidon_parser_ceiling_is_derived_from_minimum_payload_per_cardinality() -> None:
    admitted = _minimum_payload_sidon_elements(MAX_SIDON_SET_SIZE)
    rejected = tuple(str(value) for value in range(MAX_SIDON_SET_SIZE + 1))

    assert (
        _minimum_integer_sidon_result_bytes(MAX_SIDON_SET_SIZE)
        <= MAX_SIDON_RESULT_BYTES
    )
    assert (
        _minimum_integer_sidon_result_bytes(MAX_SIDON_SET_SIZE + 1)
        > MAX_SIDON_RESULT_BYTES
    )
    assert _integer_sidon_canonical_result_bytes(
        admitted
    ) == _minimum_integer_sidon_result_bytes(MAX_SIDON_SET_SIZE)
    assert MAX_SIDON_ORDERED_DIFFERENCES == MAX_SIDON_SET_SIZE * (
        MAX_SIDON_SET_SIZE - 1
    )
    IntegerSidonRequest(elements=tuple(str(value) for value in admitted))
    _require_integer_sidon_result_admission(admitted)
    with pytest.raises(ValidationError) as exc_info:
        IntegerSidonRequest(elements=rejected)
    assert exc_info.value.errors()[0]["type"] == "too_long"


def test_sidon_translated_interval_reaches_result_sensitive_admission() -> None:
    elements = tuple(range(-9, 426))
    request = IntegerSidonRequest(elements=tuple(str(value) for value in elements))
    plan = _require_integer_sidon_result_admission(elements)

    assert len(request.elements) == 435
    assert MAX_SIDON_SET_SIZE == 435
    assert plan.result_bytes == 10_481_930
    assert plan.result_bytes <= MAX_SIDON_RESULT_BYTES


def test_sidon_nonnegative_interval_at_the_ceiling_is_rejected_by_bytes() -> None:
    elements = tuple(range(MAX_SIDON_SET_SIZE))
    IntegerSidonRequest(elements=tuple(str(value) for value in elements))

    assert _integer_sidon_canonical_result_bytes(elements) > MAX_SIDON_RESULT_BYTES
    with pytest.raises(OperationDomainValidationError, match="canonical output bound"):
        _require_integer_sidon_result_admission(elements)


def test_sidon_zero_through_256_reaches_result_sensitive_admission() -> None:
    elements = tuple(range(257))
    request = IntegerSidonRequest(elements=tuple(str(value) for value in elements))

    assert len(request.elements) == 257
    assert _integer_sidon_canonical_result_bytes(elements) == 3_616_904
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
    result = decide_integer_sidon(IntegerSidonRequest(elements=("0", "1", "3")))

    assert calls == 1
    plan = captured[0]
    assert result.normalized_elements == plan.normalized_wires
    assert result.is_sidon == plan.is_sidon
    assert (
        tuple(
            (item.minuend, item.subtrahend, item.difference)
            for item in result.ordered_differences
        )
        == plan.difference_wires
    )


def test_sidon_rejects_profile_exceeding_canonical_output_bound() -> None:
    prefix = "9" * 125
    elements = tuple(f"{prefix}{value:03d}" for value in range(256))

    with pytest.raises(OperationDomainValidationError, match="canonical output bound"):
        decide_integer_sidon(IntegerSidonRequest(elements=elements))


def test_sidon_result_rejects_forged_difference_and_decision() -> None:
    result = decide_integer_sidon(IntegerSidonRequest(elements=("0", "1", "3")))
    payload = result.model_dump(mode="json")
    payload["ordered_differences"][0]["difference"] = "0"
    payload["is_sidon"] = not result.is_sidon

    with raises_code("combinatorics.sidon_invariant"):
        IntegerSidonResult.model_validate(payload)


def test_extension_request_rejects_an_unbounded_candidate_space() -> None:
    request = CyclicDifferenceSetExtensionRequest(
        base_elements=("0", "1", "2", "3", "4", "5", "6"),
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
        CyclicDifferenceSetExtensionRequest(base_elements=("0", "1"), target_order=3)
    )
    assert result.decision == "EXTENDS"
    payload = result.model_dump(mode="json")
    payload["extension"] = [0, 2, 4]
    with pytest.raises(ValidationError):
        CyclicDifferenceSetExtensionResult.model_validate(payload)
