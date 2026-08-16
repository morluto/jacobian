"""Tests for numerical semigroup operations."""

from jacobian.contracts.numerical_semigroups import (
    NumericalSemigroupSummaryRequest,
    SemigroupMembershipRequest,
)
from jacobian.domains.numerical_semigroups.operations import (
    compute_membership,
    compute_summary,
)


class TestSemigroupSummary:
    def test_semigroup_3_5(self):
        req = NumericalSemigroupSummaryRequest(generators=("3", "5"))
        result = compute_summary(req)
        assert result.frobenius_number == "7"
        assert result.multiplicity == "3"
        assert result.genus == 4
        gaps = [int(g) for g in result.gaps]
        assert gaps == [1, 2, 4, 7]
        assert result.conductor == "8"

    def test_semigroup_4_5_6(self):
        req = NumericalSemigroupSummaryRequest(generators=("4", "5", "6"))
        result = compute_summary(req)
        assert result.multiplicity == "4"
        # Frobenius number of <4,5,6> is 7
        assert result.frobenius_number == "7"

    def test_two_generators(self):
        req = NumericalSemigroupSummaryRequest(generators=("2", "3"))
        result = compute_summary(req)
        # <2,3> has Frobenius=1, gaps={1}, genus=1
        assert result.frobenius_number == "1"
        assert result.genus == 1

    def test_two_and_one_hundred_one(self):
        req = NumericalSemigroupSummaryRequest(generators=("2", "101"))
        result = compute_summary(req)
        assert result.frobenius_number == "99"
        assert result.conductor == "100"
        assert result.genus == 50

    def test_rejects_nonpositive_generators(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="positive"):
            NumericalSemigroupSummaryRequest(generators=("-1",))
        with pytest.raises(ValidationError, match="positive"):
            SemigroupMembershipRequest(generators=("0", "2"), value="4")


class TestSemigroupMembership:
    def test_in_semigroup(self):
        req = SemigroupMembershipRequest(generators=("3", "5"), value="8")
        result = compute_membership(req)
        assert result.in_semigroup is True

    def test_not_in_semigroup(self):
        req = SemigroupMembershipRequest(generators=("3", "5"), value="7")
        result = compute_membership(req)
        assert result.in_semigroup is False

    def test_zero_is_in(self):
        req = SemigroupMembershipRequest(generators=("3", "5"), value="0")
        result = compute_membership(req)
        assert result.in_semigroup is True
