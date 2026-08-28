from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._difference_set_models import (
    CyclicDifferenceMultiplicity,
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    CyclicPerfectDifferenceSetRequest,
    CyclicPerfectDifferenceSetResult,
    IntegerSidonRequest,
    IntegerSidonResult,
)
from jacobian.math.combinatorics._difference_sets import (
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
