"""Tests for numerical semigroup operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.numerical_semigroup import (
    NumericalSemigroupMembershipRequest,
    NumericalSemigroupRequest,
)
from jacobian.domains.number_theory.numerical_semigroup_operations import (
    check_numerical_semigroup_membership,
    compute_numerical_semigroup_summary,
)


def test_semigroup_3_5():
    """S = <3, 5> has Frobenius number 7, conductor 8, genus 4."""
    result = compute_numerical_semigroup_summary(
        NumericalSemigroupRequest.model_validate({"generators": [3, 5]})
    )
    assert result.frobenius_number == 7
    assert result.conductor == 8
    assert result.genus == 4
    assert result.gaps == (1, 2, 4, 7)
    assert result.multiplicity == 3
    assert result.embedding_dimension == 2


def test_semigroup_4_5_6():
    """S = <4, 5, 6> has Frobenius number 7, conductor 8."""
    result = compute_numerical_semigroup_summary(
        NumericalSemigroupRequest.model_validate({"generators": [4, 5, 6]})
    )
    assert result.frobenius_number == 7
    assert result.conductor == 8


def test_semigroup_3_5_7():
    """S = <3, 5, 7> has Frobenius number 4."""
    result = compute_numerical_semigroup_summary(
        NumericalSemigroupRequest.model_validate({"generators": [3, 5, 7]})
    )
    assert result.frobenius_number == 4
    assert result.gaps == (1, 2, 4)


def test_membership_present():
    """8 is in <3, 5>."""
    result = check_numerical_semigroup_membership(
        NumericalSemigroupMembershipRequest.model_validate({
            "generators": [3, 5],
            "element": 8,
        })
    )
    assert result.is_member is True


def test_membership_absent():
    """1 is not in <3, 5>."""
    result = check_numerical_semigroup_membership(
        NumericalSemigroupMembershipRequest.model_validate({
            "generators": [3, 5],
            "element": 1,
        })
    )
    assert result.is_member is False


def test_gcd_one_required():
    """Generators with gcd > 1 should fail."""
    with pytest.raises(ValidationError, match="gcd 1"):
        NumericalSemigroupRequest.model_validate({"generators": [4, 6]})


def test_positive_generators_required():
    """Non-positive generators should fail."""
    with pytest.raises(ValidationError, match="positive"):
        NumericalSemigroupRequest.model_validate({"generators": [0, 1]})


def test_operations_discoverable():
    """Both operations should be discoverable via the factory."""
    from jacobian.domains.number_theory import number_theory_operations

    ops = number_theory_operations()
    op_ids = [op.operation_id for op in ops]
    assert "number_theory.numerical_semigroup.summary.compute" in op_ids
    assert "number_theory.numerical_semigroup.membership.check" in op_ids
