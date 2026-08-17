from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics._models import (
    CyclicDifferenceMultiplicity,
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    CyclicPerfectDifferenceSetResult,
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


def test_pds_result_rejects_a_forged_self_consistent_profile() -> None:
    """A producer regression cannot pass a forged multiplicity profile.

    Residues ``(0, 1, 2)`` modulo 7 repeat differences 1 and 6 and omit 3 and
    4, but a forged profile claiming multiplicity 1 for every nonzero residue
    with empty missing/repeated lists and ``is_perfect=True`` is internally
    self-consistent under the old shape-only validator. The authoritative
    result model must recompute multiplicities from the residues.
    """
    forged_profile = tuple(
        CyclicDifferenceMultiplicity(residue=residue, multiplicity=1)
        for residue in range(1, 7)
    )
    with pytest.raises(ValidationError, match="derived from the residues"):
        CyclicPerfectDifferenceSetResult(
            semantics_version="cyclic-perfect-difference-set.v1",
            modulus=7,
            normalized_residues=(0, 1, 2),
            order=3,
            expected_modulus=7,
            difference_multiplicities=forged_profile,
            missing_residues=(),
            repeated_residues=(),
            is_perfect=True,
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
        semantics_version="cyclic-perfect-difference-set.v1",
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
