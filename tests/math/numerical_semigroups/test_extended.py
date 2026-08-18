"""Tests for extended numerical semigroup factorization operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.numerical_semigroups._models import (
    BettiElementsRequest,
    CatenaryDegreeRequest,
    DeltaSetRequest,
    ElasticityRequest,
    ElementCatenaryDegreeRequest,
    ElementDeltaSetRequest,
    ElementElasticityRequest,
    FactorizationComputeRequest,
    FactorizationDistanceRequest,
    FactorizationGraphComputeRequest,
    FactorizationLengthsComputeRequest,
    MinimalPresentationRequest,
    PresentationBinomialsRequest,
)
from jacobian.math.numerical_semigroups._operations import (
    compute_betti_elements,
    compute_catenary_degree,
    compute_delta_set,
    compute_elasticity,
    compute_element_catenary_degree,
    compute_element_delta_set,
    compute_element_elasticity,
    compute_factorization_distance,
    compute_factorization_graph,
    compute_factorization_lengths,
    compute_factorizations,
    compute_minimal_presentation,
    compute_presentation_binomials,
)


class TestFactorizations:
    def test_factorizations_15_in_3_5(self):
        req = FactorizationComputeRequest(generators=("3", "5"), value="15")
        result = compute_factorizations(req)
        assert result.value == "15"
        assert result.minimal_generators == ("3", "5")
        assert set(result.factorizations) == {(5, 0), (0, 3)}

    def test_factorizations_12_in_3_5(self):
        req = FactorizationComputeRequest(generators=("3", "5"), value="12")
        result = compute_factorizations(req)
        assert set(result.factorizations) == {(4, 0)}

    def test_factorizations_zero(self):
        req = FactorizationComputeRequest(generators=("3", "5"), value="0")
        result = compute_factorizations(req)
        assert result.factorizations == ((0, 0),)

    def test_factorizations_non_member(self):
        req = FactorizationComputeRequest(generators=("3", "5"), value="7")
        result = compute_factorizations(req)
        assert result.factorizations == ()

    def test_factorizations_rejects_nonpositive_generators(self):
        with pytest.raises(ValidationError, match="positive"):
            FactorizationComputeRequest(generators=("0", "5"), value="10")

    def test_factorizations_non_minimal_generators(self):
        """Generators include a non-minimal element (e.g., 8 in <3,5>)."""
        req = FactorizationComputeRequest(generators=("3", "5", "8"), value="15")
        result = compute_factorizations(req)
        assert result.minimal_generators == ("3", "5")
        assert set(result.factorizations) == {(5, 0), (0, 3)}


class TestFactorizationLengths:
    def test_lengths_15_in_3_5(self):
        req = FactorizationLengthsComputeRequest(
            generators=("3", "5"), value="15"
        )
        result = compute_factorization_lengths(req)
        assert result.lengths == (3, 5)

    def test_lengths_single(self):
        req = FactorizationLengthsComputeRequest(
            generators=("3", "5"), value="12"
        )
        result = compute_factorization_lengths(req)
        assert result.lengths == (4,)

    def test_lengths_empty_non_member(self):
        req = FactorizationLengthsComputeRequest(
            generators=("3", "5"), value="7"
        )
        result = compute_factorization_lengths(req)
        assert result.lengths == ()

    def test_lengths_consecutive_for_nugget(self):
        """<4,6,9>: factorizations of 36 have lengths 4..9 (consecutive)."""
        req = FactorizationLengthsComputeRequest(
            generators=("4", "6", "9"), value="36"
        )
        result = compute_factorization_lengths(req)
        assert result.lengths == (4, 5, 6, 7, 8, 9)


class TestFactorizationDistance:
    def test_distance_15_in_3_5(self):
        req = FactorizationDistanceRequest(
            generators=("3", "5"), value="15", first=(5, 0), second=(0, 3)
        )
        result = compute_factorization_distance(req)
        assert result.distance == 5
        assert result.first_length == 5
        assert result.second_length == 3

    def test_distance_identical_factorization(self):
        req = FactorizationDistanceRequest(
            generators=("3", "5"), value="15", first=(5, 0), second=(5, 0)
        )
        result = compute_factorization_distance(req)
        assert result.distance == 0

    def test_distance_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError, match="equal coordinate length"):
            FactorizationDistanceRequest(
                generators=("3", "5"), value="15", first=(5, 0, 0), second=(0, 3)
            )

    def test_distance_rejects_negative_coordinates(self):
        with pytest.raises(ValidationError, match="non-negative"):
            FactorizationDistanceRequest(
                generators=("3", "5"), value="15", first=(-1, 0), second=(0, 3)
            )


class TestFactorizationGraph:
    def test_graph_15_in_3_5_disconnected(self):
        req = FactorizationGraphComputeRequest(
            generators=("3", "5"), value="15"
        )
        result = compute_factorization_graph(req)
        assert not result.is_connected
        assert len(result.connected_components) == 2
        assert len(result.factorizations) == 2

    def test_graph_12_in_3_5_connected(self):
        req = FactorizationGraphComputeRequest(
            generators=("3", "5"), value="12"
        )
        result = compute_factorization_graph(req)
        assert result.is_connected
        assert len(result.connected_components) == 1

    def test_graph_edges(self):
        req = FactorizationGraphComputeRequest(
            generators=("4", "6", "9"), value="12"
        )
        result = compute_factorization_graph(req)
        # 12 = 3*4 + 0*6 + 0*9 = (3,0,0)
        # 12 = 0*4 + 2*6 + 0*9 = (0,2,0)
        # gcd = (0,0,0) so not connected
        assert not result.is_connected


class TestElementDeltaSet:
    def test_delta_set_15_in_3_5(self):
        req = ElementDeltaSetRequest(generators=("3", "5"), value="15")
        result = compute_element_delta_set(req)
        assert result.delta_set == (2,)

    def test_delta_set_36_in_4_6_9(self):
        """Lengths of 36 in <4,6,9> are {4,5,6,7,8,9} -> delta = {1,1,1,1,1}."""
        req = ElementDeltaSetRequest(generators=("4", "6", "9"), value="36")
        result = compute_element_delta_set(req)
        assert result.delta_set == (1, 1, 1, 1, 1)

    def test_delta_set_single_factorization(self):
        req = ElementDeltaSetRequest(generators=("3", "5"), value="12")
        result = compute_element_delta_set(req)
        assert result.delta_set == ()


class TestElementElasticity:
    def test_elasticity_15_in_3_5(self):
        req = ElementElasticityRequest(generators=("3", "5"), value="15")
        result = compute_element_elasticity(req)
        assert result.elasticity == "5/3"

    def test_elasticity_single_factorization(self):
        req = ElementElasticityRequest(generators=("3", "5"), value="12")
        result = compute_element_elasticity(req)
        assert result.elasticity == "1"

    def test_elasticity_36_in_4_6_9(self):
        req = ElementElasticityRequest(generators=("4", "6", "9"), value="36")
        result = compute_element_elasticity(req)
        assert result.elasticity == "9/4"


class TestElementCatenaryDegree:
    def test_catenary_15_in_3_5(self):
        req = ElementCatenaryDegreeRequest(generators=("3", "5"), value="15")
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 5

    def test_catenary_connected_graph(self):
        """If the factorization graph is connected, catenary degree is 0."""
        req = ElementCatenaryDegreeRequest(generators=("3", "5"), value="12")
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 0

    def test_catenary_single_factorization(self):
        req = ElementCatenaryDegreeRequest(
            generators=("3", "5"), value="3"
        )
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 0


class TestBettiElements:
    def test_betti_3_5(self):
        req = BettiElementsRequest(generators=("3", "5"))
        result = compute_betti_elements(req)
        assert result.betti_elements == ("15",)

    def test_betti_4_6_9(self):
        req = BettiElementsRequest(generators=("4", "6", "9"))
        result = compute_betti_elements(req)
        assert result.betti_elements == ("12", "18")

    def test_betti_2_3(self):
        req = BettiElementsRequest(generators=("2", "3"))
        result = compute_betti_elements(req)
        # <2,3>: Betti element should include 6=lcm(2,3)
        assert "6" in result.betti_elements


class TestMinimalPresentation:
    def test_presentation_3_5(self):
        req = MinimalPresentationRequest(generators=("3", "5"))
        result = compute_minimal_presentation(req)
        assert result.betti_elements == ("15",)
        assert len(result.relations) == 1
        assert result.relations[0].first == (5, 0)
        assert result.relations[0].second == (0, 3)

    def test_presentation_4_6_9(self):
        req = MinimalPresentationRequest(generators=("4", "6", "9"))
        result = compute_minimal_presentation(req)
        assert result.betti_elements == ("12", "18")
        assert len(result.relations) >= 2


class TestPresentationBinomials:
    def test_binomials_3_5(self):
        req = PresentationBinomialsRequest(
            generators=("3", "5"),
            relations=[{"first": [5, 0], "second": [0, 3]}],
        )
        result = compute_presentation_binomials(req)
        assert len(result.binomials) == 1
        b = result.binomials[0]
        assert b.left_coefficient == "5"
        assert b.left_exponents == (5, 0)
        assert b.right_coefficient == "3"
        assert b.right_exponents == (0, 3)

    def test_binomials_4_6_9(self):
        req = PresentationBinomialsRequest(
            generators=("4", "6", "9"),
            relations=[
                {"first": [3, 0, 0], "second": [0, 2, 0]},
                {"first": [0, 3, 0], "second": [0, 0, 2]},
            ],
        )
        result = compute_presentation_binomials(req)
        assert len(result.binomials) == 2

    def test_binomials_rejects_empty_relations(self):
        with pytest.raises(ValidationError):
            PresentationBinomialsRequest(
                generators=("3", "5"),
                relations=[],
            )


class TestGlobalDeltaSet:
    def test_delta_set_3_5(self):
        req = DeltaSetRequest(generators=("3", "5"))
        result = compute_delta_set(req)
        assert result.delta_set == (2,)

    def test_delta_set_4_6_9(self):
        req = DeltaSetRequest(generators=("4", "6", "9"))
        result = compute_delta_set(req)
        assert result.delta_set == (1,)


class TestGlobalElasticity:
    def test_elasticity_3_5(self):
        req = ElasticityRequest(generators=("3", "5"))
        result = compute_elasticity(req)
        assert result.elasticity == "5/3"

    def test_elasticity_4_6_9(self):
        req = ElasticityRequest(generators=("4", "6", "9"))
        result = compute_elasticity(req)
        assert result.elasticity == "9/4"

    def test_elasticity_2_3(self):
        req = ElasticityRequest(generators=("2", "3"))
        result = compute_elasticity(req)
        assert result.elasticity == "3/2"


class TestGlobalCatenaryDegree:
    def test_catenary_3_5(self):
        req = CatenaryDegreeRequest(generators=("3", "5"))
        result = compute_catenary_degree(req)
        assert result.catenary_degree == 5

    def test_catenary_4_6_9(self):
        req = CatenaryDegreeRequest(generators=("4", "6", "9"))
        result = compute_catenary_degree(req)
        assert result.catenary_degree == 3
