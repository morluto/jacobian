"""Tests for numerical semigroup operations."""

import pytest
from pydantic import ValidationError
from tests.math.number_theory.numerical_semigroups._support import (
    operation_domain_error,
)

from jacobian.math.number_theory.numerical_semigroups._models import (
    MAX_ELEMENT,
    MAX_GENERATOR,
    MAX_GENERATORS,
)
from jacobian.math.number_theory.numerical_semigroups._summary_models import (
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
    SemigroupMembershipRequest,
)
from jacobian.math.number_theory.numerical_semigroups._summary_operations import (
    compute_membership,
    compute_summary,
)


class TestSemigroupSummary:
    def test_summary_request_schema_describes_its_admission_envelope(self) -> None:
        generators = NumericalSemigroupSummaryRequest.model_json_schema()["properties"][
            "generators"
        ]

        assert str(MAX_GENERATOR) in generators["description"]
        assert "gcd 1" in generators["description"]
        assert generators["maxItems"] == MAX_GENERATORS

    def test_semigroup_3_5(self) -> None:
        req = NumericalSemigroupSummaryRequest(generators=("3", "5"))
        result = compute_summary(req)
        assert result.frobenius_number == "7"
        assert result.multiplicity == "3"
        assert result.genus == 4
        gaps = [int(g) for g in result.gaps]
        assert gaps == [1, 2, 4, 7]
        assert result.conductor == "8"

    def test_semigroup_4_5_6(self) -> None:
        req = NumericalSemigroupSummaryRequest(generators=("4", "5", "6"))
        result = compute_summary(req)
        assert result.multiplicity == "4"
        # Frobenius number of <4,5,6> is 7
        assert result.frobenius_number == "7"

    def test_two_generators(self) -> None:
        req = NumericalSemigroupSummaryRequest(generators=("2", "3"))
        result = compute_summary(req)
        # <2,3> has Frobenius=1, gaps={1}, genus=1
        assert result.frobenius_number == "1"
        assert result.genus == 1

    def test_two_and_one_hundred_one(self) -> None:
        req = NumericalSemigroupSummaryRequest(generators=("2", "101"))
        result = compute_summary(req)
        assert result.frobenius_number == "99"
        assert result.conductor == "100"
        assert result.genus == 50

    def test_rejects_nonpositive_generators(self) -> None:
        with operation_domain_error():
            compute_summary(NumericalSemigroupSummaryRequest(generators=("-1",)))
        with operation_domain_error():
            compute_membership(
                SemigroupMembershipRequest(generators=("0", "2"), value="4")
            )

    @pytest.mark.parametrize("axis", [(), tuple(map(str, range(30, 51)))])
    def test_summary_result_rejects_empty_or_overlong_minimal_axis(
        self, axis: tuple[str, ...]
    ) -> None:
        with pytest.raises(ValidationError):
            NumericalSemigroupSummaryResult(
                minimal_generators=axis,
                multiplicity="3",
                embedding_dimension=2,
                frobenius_number="7",
                conductor="8",
                genus=4,
                gaps=("1", "2", "4", "7"),
            )


class TestSemigroupMembership:
    def test_membership_request_schema_describes_its_admission_envelope(
        self,
    ) -> None:
        properties = SemigroupMembershipRequest.model_json_schema()["properties"]

        assert str(MAX_GENERATOR) in properties["generators"]["description"]
        assert "gcd 1" in properties["generators"]["description"]
        assert properties["generators"]["maxItems"] == MAX_GENERATORS
        assert str(MAX_ELEMENT) in properties["value"]["description"]

    def test_in_semigroup(self) -> None:
        req = SemigroupMembershipRequest(generators=("3", "5"), value="8")
        result = compute_membership(req)
        assert result.in_semigroup is True

    def test_not_in_semigroup(self) -> None:
        req = SemigroupMembershipRequest(generators=("3", "5"), value="7")
        result = compute_membership(req)
        assert result.in_semigroup is False

    def test_zero_is_in(self) -> None:
        req = SemigroupMembershipRequest(generators=("3", "5"), value="0")
        result = compute_membership(req)
        assert result.in_semigroup is True

    def test_free_axis_membership_does_not_allocate_to_the_element_value(
        self,
    ) -> None:
        value = str(MAX_ELEMENT + 1)
        result = compute_membership(
            SemigroupMembershipRequest(
                generators=("1", str(MAX_GENERATOR + 1)), value=value
            )
        )
        assert result.value == value
        assert result.in_semigroup is True
