from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.cyclic_sumset_profile._models import (
    CyclicSumsetRequest,
)
from jacobian.math.combinatorics.additive.cyclic_sumset_profile.operations import (
    compute_cyclic_sumset_profile,
)


def test_simple() -> None:
    result = compute_cyclic_sumset_profile(5, (0, 1), (0, 2))
    entries = {e.residue: e.count for e in result.entries}
    # 0+0=0, 0+2=2, 1+0=1, 1+2=3
    assert entries == {0: 1, 1: 1, 2: 1, 3: 1}


def test_empty() -> None:
    result = compute_cyclic_sumset_profile(5, (), (0, 1))
    assert result.support_cardinality == 0


def test_modular_wraparound() -> None:
    result = compute_cyclic_sumset_profile(5, (3,), (4,))
    entries = {e.residue: e.count for e in result.entries}
    # 3+4=7=2 mod 5
    assert entries == {2: 1}


def test_multiple_representations() -> None:
    result = compute_cyclic_sumset_profile(6, (0, 2, 4), (0, 2, 4))
    entries = {e.residue: e.count for e in result.entries}
    # Check total representations
    total = sum(entries.values())
    assert total == 9  # 3 * 3


def test_result_preserves_source() -> None:
    result = compute_cyclic_sumset_profile(7, (0, 1), (2, 3))
    assert result.modulus == 7
    assert result.left == (0, 1)
    assert result.right == (2, 3)


def test_nonpositive_modulus_is_rejected_before_arithmetic() -> None:
    with pytest.raises(OperationDomainValidationError, match="must be positive"):
        compute_cyclic_sumset_profile(0, (0,), (0,))


def test_noncanonical_residue_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="canonical residues"):
        compute_cyclic_sumset_profile(5, (5,), (0,))


@pytest.mark.parametrize(
    ("left", "right"),
    [((0, 0), (1,)), ((0,), (1, 1))],
)
def test_duplicate_subset_elements_are_rejected(
    left: tuple[int, ...], right: tuple[int, ...]
) -> None:
    with pytest.raises(OperationDomainValidationError, match="distinct residues"):
        compute_cyclic_sumset_profile(5, left, right)


def test_request_rejects_duplicate_subset_elements() -> None:
    with pytest.raises(ValidationError, match="distinct residues"):
        CyclicSumsetRequest(modulus=2, left=(0, 0), right=(0,))


def test_native_operation_is_exported_by_owner_package() -> None:
    from jacobian.math.combinatorics.additive.cyclic_sumset_profile import (
        compute_cyclic_sumset_profile as exported,
    )

    assert exported is compute_cyclic_sumset_profile


def test_modulus_must_fit_the_interoperable_integer_carrier() -> None:
    left = tuple(10**99 + index for index in range(50_000))
    right = (0,)

    with pytest.raises(OperationDomainValidationError, match="interoperable JSON"):
        compute_cyclic_sumset_profile(10**101, left, right)
    with pytest.raises(ValidationError):
        CyclicSumsetRequest(modulus=10**101, left=left, right=right)
