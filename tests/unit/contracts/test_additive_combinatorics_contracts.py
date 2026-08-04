from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.combinatorics import (
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    IntegerSidonRequest,
    IntegerSidonResult,
)


def test_sidon_request_rejects_duplicate_integer_elements() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        IntegerSidonRequest(elements=("1", "2", "1"))


def test_sidon_result_requires_the_complete_ordered_profile() -> None:
    with pytest.raises(ValidationError, match="every distinct ordered pair"):
        IntegerSidonResult(
            semantics_version="integer-sidon.ordered-differences.v1",
            normalized_elements=("1", "2", "4"),
            ordered_differences=(),
            is_sidon=True,
        )


def test_extension_request_rejects_an_unbounded_candidate_space() -> None:
    with pytest.raises(ValidationError, match="candidate space"):
        CyclicDifferenceSetExtensionRequest(
            base_elements=("0", "1", "2", "3", "4", "5", "6"),
            target_order=10,
        )


def test_negative_extension_result_binds_exact_candidate_count() -> None:
    with pytest.raises(ValidationError, match="exact combination space"):
        CyclicDifferenceSetExtensionResult(
            semantics_version="cyclic-pds-extension.fixed-order.v1",
            target_order=6,
            modulus=31,
            base_residues=(1, 2, 4, 8, 13),
            candidate_space_size=25,
            decision="DOES_NOT_EXTEND",
            extension=(),
            coverage="ALL_CANDIDATES",
        )


def test_positive_extension_result_rejects_residue_outside_modulus() -> None:
    with pytest.raises(ValidationError, match="derived modulus"):
        CyclicDifferenceSetExtensionResult(
            semantics_version="cyclic-pds-extension.fixed-order.v1",
            target_order=3,
            modulus=7,
            base_residues=(0, 1),
            candidate_space_size=5,
            decision="EXTENDS",
            extension=(0, 1, 7),
            coverage="WITNESS",
        )
