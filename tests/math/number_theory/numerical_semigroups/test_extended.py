"""Tests for extended numerical semigroup factorization operations."""

import pytest
from pydantic import BaseModel, ValidationError
from tests.math.number_theory.numerical_semigroups._support import (
    numerical_semigroup_error,
    operation_domain_error,
)

from jacobian.math.number_theory.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)
from jacobian.math.number_theory.numerical_semigroups._element_invariant_operations import (
    compute_element_catenary_degree,
    compute_element_delta_set,
    compute_element_elasticity,
)
from jacobian.math.number_theory.numerical_semigroups._factorization_models import (
    FactorizationComputeRequest,
    FactorizationComputeResult,
    FactorizationDistanceRequest,
    FactorizationGraphComputeRequest,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeRequest,
    FactorizationLengthsComputeResult,
)
from jacobian.math.number_theory.numerical_semigroups._factorization_operations import (
    compute_factorization_distance,
    compute_factorization_graph,
    compute_factorization_lengths,
    compute_factorizations,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_models import (
    BettiElementsRequest,
    BettiElementsResult,
    CatenaryDegreeRequest,
    CatenaryDegreeResult,
    DeltaSetRequest,
    DeltaSetResult,
    ElasticityRequest,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_operations import (
    compute_betti_elements,
    compute_catenary_degree,
    compute_delta_set,
    compute_elasticity,
)
from jacobian.math.number_theory.numerical_semigroups._models import (
    MAX_ELEMENT,
    MAX_GENERATOR,
)
from jacobian.math.number_theory.numerical_semigroups._presentation_models import (
    MinimalPresentationRequest,
    MinimalPresentationResult,
    PresentationBinomialsRequest,
    PresentationBinomialsResult,
)
from jacobian.math.number_theory.numerical_semigroups._presentation_operations import (
    compute_minimal_presentation,
    compute_presentation_binomials,
)


class TestFactorizations:
    def test_factorizations_15_in_3_5(self) -> None:
        req = FactorizationComputeRequest(generators=("3", "5"), value="15")
        result = compute_factorizations(req)
        assert result.value == "15"
        assert result.minimal_generators == ("3", "5")
        assert set(result.factorizations) == {(5, 0), (0, 3)}

    def test_factorizations_12_in_3_5(self) -> None:
        req = FactorizationComputeRequest(generators=("3", "5"), value="12")
        result = compute_factorizations(req)
        assert set(result.factorizations) == {(4, 0)}

    def test_factorizations_zero(self) -> None:
        req = FactorizationComputeRequest(generators=("3", "5"), value="0")
        result = compute_factorizations(req)
        assert result.factorizations == ((0, 0),)

    def test_factorizations_non_member(self) -> None:
        req = FactorizationComputeRequest(generators=("3", "5"), value="7")
        result = compute_factorizations(req)
        assert result.factorizations == ()

    def test_factorizations_rejects_nonpositive_generators(self) -> None:
        with operation_domain_error():
            compute_factorizations(
                FactorizationComputeRequest(generators=("0", "5"), value="10")
            )

    def test_factorizations_normalize_redundant_permuted_generators(self) -> None:
        """Factorizations always use the canonical minimal-generator axis."""
        result = compute_factorizations(
            FactorizationComputeRequest(generators=("8", "5", "3"), value="15")
        )

        assert result.minimal_generators == ("3", "5")
        assert result.factorizations == ((0, 3), (5, 0))

    def test_factorization_result_rejects_a_redundant_coordinate_axis(self) -> None:
        with numerical_semigroup_error():
            FactorizationComputeResult(
                value="15",
                minimal_generators=("3", "5", "8"),
                in_semigroup=True,
                factorizations=((0, 3, 0), (5, 0, 0)),
            )

    def test_factorization_materialization_is_complete_past_old_silent_cap(
        self,
    ) -> None:
        generators = ("6", "7", "8", "9", "10", "11")
        result = compute_factorizations(
            FactorizationComputeRequest(generators=generators, value="200")
        )
        counts = [0] * 201
        counts[0] = 1
        for generator in map(int, generators):
            for value in range(generator, 201):
                counts[value] += counts[value - generator]
        assert counts[200] == 14_506
        assert len(result.factorizations) == counts[200]
        assert result.in_semigroup


@pytest.mark.parametrize(
    ("result_model", "payload"),
    [
        (
            FactorizationComputeResult,
            {"value": "15", "in_semigroup": True, "factorizations": ((0, 3, 0),)},
        ),
        (
            FactorizationLengthsComputeResult,
            {"value": "18", "in_semigroup": True, "lengths": (3, 4, 5, 6)},
        ),
        (
            FactorizationGraphComputeResult,
            {
                "value": "15",
                "in_semigroup": True,
                "factorizations": ((0, 3, 0),),
                "edges": (),
                "connected_components": ((0,),),
                "is_connected": True,
            },
        ),
        (
            ElementDeltaSetResult,
            {
                "value": "18",
                "factorization_lengths": (3, 4, 5, 6),
                "delta_set": (1,),
            },
        ),
        (
            ElementElasticityResult,
            {
                "value": "18",
                "minimum_length": 3,
                "maximum_length": 6,
                "elasticity": "2",
            },
        ),
        (
            ElementCatenaryDegreeResult,
            {"value": "18", "factorization_count": 5, "catenary_degree": 3},
        ),
        (
            BettiElementsResult,
            {
                "apery_set": ("0", "10", "5"),
                "candidate_count": 6,
                "betti_elements": ("15",),
            },
        ),
        (
            MinimalPresentationResult,
            {
                "betti_elements": ("15",),
                "relations": ({"first": (5, 0, 0), "second": (0, 3, 0)},),
            },
        ),
        (
            PresentationBinomialsResult,
            {
                "binomials": (
                    {
                        "left_exponents": (5, 0, 0),
                        "right_exponents": (0, 3, 0),
                    },
                )
            },
        ),
        (
            DeltaSetResult,
            {"delta_set": (1,), "periodicity_bound": 45, "checked_through": 50},
        ),
        (
            CatenaryDegreeResult,
            {
                "catenary_degree": 3,
                "betti_degrees": ({"betti_element": "15", "catenary_degree": 3},),
                "witness_betti_elements": ("15",),
            },
        ),
    ],
)
def test_result_rejects_redundant_minimal_generator_axis(
    result_model: type[BaseModel], payload: dict[str, object]
) -> None:
    with numerical_semigroup_error():
        result_model.model_validate({"minimal_generators": ("3", "5", "6"), **payload})


@pytest.mark.parametrize("axis", [(), tuple(map(str, range(30, 51)))])
@pytest.mark.parametrize(
    ("result_model", "payload"),
    [
        (
            FactorizationComputeResult,
            {"value": "-1", "in_semigroup": False, "factorizations": ()},
        ),
        (
            FactorizationLengthsComputeResult,
            {"value": "-1", "in_semigroup": False, "lengths": ()},
        ),
        (
            FactorizationGraphComputeResult,
            {
                "value": "-1",
                "in_semigroup": False,
                "factorizations": (),
                "edges": (),
                "connected_components": (),
                "is_connected": True,
            },
        ),
        (
            ElementDeltaSetResult,
            {"value": "0", "factorization_lengths": (0,), "delta_set": ()},
        ),
        (
            ElementElasticityResult,
            {
                "value": "30",
                "minimum_length": 1,
                "maximum_length": 1,
                "elasticity": "1",
            },
        ),
        (
            ElementCatenaryDegreeResult,
            {"value": "30", "factorization_count": 1, "catenary_degree": 0},
        ),
        (
            BettiElementsResult,
            {"apery_set": (), "candidate_count": 0, "betti_elements": ()},
        ),
        (
            MinimalPresentationResult,
            {"betti_elements": (), "relations": ()},
        ),
        (PresentationBinomialsResult, {"binomials": ()}),
        (
            DeltaSetResult,
            {"delta_set": (), "periodicity_bound": 0, "checked_through": 0},
        ),
        (
            CatenaryDegreeResult,
            {
                "catenary_degree": 0,
                "betti_degrees": (),
                "witness_betti_elements": (),
            },
        ),
    ],
)
def test_result_rejects_empty_or_overlong_minimal_generator_axis(
    axis: tuple[str, ...],
    result_model: type[BaseModel],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        result_model.model_validate({"minimal_generators": axis, **payload})


class TestFactorizationLengths:
    def test_lengths_15_in_3_5(self) -> None:
        req = FactorizationLengthsComputeRequest(generators=("3", "5"), value="15")
        result = compute_factorization_lengths(req)
        assert result.lengths == (3, 5)

    def test_lengths_single(self) -> None:
        req = FactorizationLengthsComputeRequest(generators=("3", "5"), value="12")
        result = compute_factorization_lengths(req)
        assert result.lengths == (4,)

    def test_lengths_empty_non_member(self) -> None:
        req = FactorizationLengthsComputeRequest(generators=("3", "5"), value="7")
        result = compute_factorization_lengths(req)
        assert result.lengths == ()

    def test_lengths_consecutive_for_nugget(self) -> None:
        """<4,6,9>: factorizations of 36 have lengths 4..9 (consecutive)."""
        req = FactorizationLengthsComputeRequest(generators=("4", "6", "9"), value="36")
        result = compute_factorization_lengths(req)
        assert result.lengths == (4, 5, 6, 7, 8, 9)


class TestFactorizationDistance:
    def test_distance_15_in_3_5(self) -> None:
        req = FactorizationDistanceRequest(
            generators=("3", "5"), value="15", first=(5, 0), second=(0, 3)
        )
        result = compute_factorization_distance(req)
        assert result.distance == 5
        assert result.first_length == 5
        assert result.second_length == 3

    def test_distance_identical_factorization(self) -> None:
        req = FactorizationDistanceRequest(
            generators=("3", "5"), value="15", first=(5, 0), second=(5, 0)
        )
        result = compute_factorization_distance(req)
        assert result.distance == 0

    def test_distance_normalizes_the_generator_presentation_not_coordinates(
        self,
    ) -> None:
        result = compute_factorization_distance(
            FactorizationDistanceRequest(
                generators=("8", "5", "3"),
                value="15",
                first=(5, 0),
                second=(0, 3),
            )
        )

        assert result.distance == 5

    def test_distance_rejects_mismatched_lengths(self) -> None:
        with operation_domain_error():
            compute_factorization_distance(
                FactorizationDistanceRequest(
                    generators=("3", "5"),
                    value="15",
                    first=(5, 0, 0),
                    second=(0, 3),
                )
            )

    def test_distance_rejects_negative_coordinates(self) -> None:
        with operation_domain_error():
            compute_factorization_distance(
                FactorizationDistanceRequest(
                    generators=("3", "5"),
                    value="15",
                    first=(-1, 0),
                    second=(0, 3),
                )
            )

    def test_distance_rejects_vectors_for_a_different_element(self) -> None:
        with operation_domain_error():
            compute_factorization_distance(
                FactorizationDistanceRequest(
                    generators=("3", "5"),
                    value="15",
                    first=(4, 0),
                    second=(0, 3),
                )
            )


class TestFactorizationGraph:
    def test_graph_normalizes_redundant_generators(self) -> None:
        result = compute_factorization_graph(
            FactorizationGraphComputeRequest(generators=("8", "3", "5"), value="15")
        )

        assert result.minimal_generators == ("3", "5")
        assert result.factorizations == ((0, 3), (5, 0))

    def test_graph_result_rejects_a_redundant_coordinate_axis(self) -> None:
        with numerical_semigroup_error():
            FactorizationGraphComputeResult(
                value="15",
                minimal_generators=("3", "5", "8"),
                in_semigroup=True,
                factorizations=((0, 3, 0), (5, 0, 0)),
                edges=(),
                connected_components=((0,), (1,)),
                is_connected=False,
            )

    def test_graph_15_in_3_5_disconnected(self) -> None:
        req = FactorizationGraphComputeRequest(generators=("3", "5"), value="15")
        result = compute_factorization_graph(req)
        assert not result.is_connected
        assert len(result.connected_components) == 2
        assert len(result.factorizations) == 2

    def test_graph_12_in_3_5_connected(self) -> None:
        req = FactorizationGraphComputeRequest(generators=("3", "5"), value="12")
        result = compute_factorization_graph(req)
        assert result.is_connected
        assert len(result.connected_components) == 1

    def test_graph_edges(self) -> None:
        req = FactorizationGraphComputeRequest(generators=("4", "6", "9"), value="12")
        result = compute_factorization_graph(req)
        # 12 = 3*4 + 0*6 + 0*9 = (3,0,0)
        # 12 = 0*4 + 2*6 + 0*9 = (0,2,0)
        # gcd = (0,0,0) so not connected
        assert not result.is_connected


class TestElementDeltaSet:
    def test_delta_set_15_in_3_5(self) -> None:
        req = ElementDeltaSetRequest(generators=("3", "5"), value="15")
        result = compute_element_delta_set(req)
        assert result.delta_set == (2,)

    def test_delta_set_36_in_4_6_9(self) -> None:
        """A delta set contains distinct successive differences."""
        req = ElementDeltaSetRequest(generators=("4", "6", "9"), value="36")
        result = compute_element_delta_set(req)
        assert result.delta_set == (1,)

    def test_delta_set_single_factorization(self) -> None:
        req = ElementDeltaSetRequest(generators=("3", "5"), value="12")
        result = compute_element_delta_set(req)
        assert result.delta_set == ()


class TestElementElasticity:
    def test_elasticity_15_in_3_5(self) -> None:
        req = ElementElasticityRequest(generators=("3", "5"), value="15")
        result = compute_element_elasticity(req)
        assert result.elasticity == "5/3"

    def test_elasticity_single_factorization(self) -> None:
        req = ElementElasticityRequest(generators=("3", "5"), value="12")
        result = compute_element_elasticity(req)
        assert result.elasticity == "1"

    def test_elasticity_36_in_4_6_9(self) -> None:
        req = ElementElasticityRequest(generators=("4", "6", "9"), value="36")
        result = compute_element_elasticity(req)
        assert result.elasticity == "9/4"


class TestElementCatenaryDegree:
    def test_catenary_15_in_3_5(self) -> None:
        req = ElementCatenaryDegreeRequest(generators=("3", "5"), value="15")
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 5

    def test_catenary_connected_graph(self) -> None:
        """R-connected does not mean catenary degree zero."""
        req = ElementCatenaryDegreeRequest(generators=("3", "5"), value="18")
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 5

    def test_catenary_single_factorization(self) -> None:
        req = ElementCatenaryDegreeRequest(generators=("3", "5"), value="3")
        result = compute_element_catenary_degree(req)
        assert result.catenary_degree == 0


class TestBettiElements:
    def test_betti_3_5(self) -> None:
        req = BettiElementsRequest(generators=("3", "5"))
        result = compute_betti_elements(req)
        assert result.betti_elements == ("15",)

    def test_betti_4_6_9(self) -> None:
        req = BettiElementsRequest(generators=("4", "6", "9"))
        result = compute_betti_elements(req)
        assert result.betti_elements == ("12", "18")

    def test_betti_2_3(self) -> None:
        req = BettiElementsRequest(generators=("2", "3"))
        result = compute_betti_elements(req)
        # <2,3>: Betti element should include 6=lcm(2,3)
        assert "6" in result.betti_elements

    def test_betti_beyond_former_heuristic_cap(self) -> None:
        result = compute_betti_elements(BettiElementsRequest(generators=("101", "103")))
        assert result.betti_elements == ("10403",)
        assert result.apery_set[0] == "0"
        assert result.candidate_count == 200

    def test_known_numericalsgps_example(self) -> None:
        result = compute_betti_elements(
            BettiElementsRequest(generators=("3", "5", "7"))
        )
        assert result.betti_elements == ("10", "12", "14")
        presentation = compute_minimal_presentation(
            MinimalPresentationRequest(generators=("3", "5", "7"))
        )
        assert {
            frozenset((relation.first, relation.second))
            for relation in presentation.relations
        } == {
            frozenset(((0, 0, 2), (3, 1, 0))),
            frozenset(((0, 1, 1), (4, 0, 0))),
            frozenset(((0, 2, 0), (1, 0, 1))),
        }


class TestMinimalPresentation:
    def test_presentation_normalizes_redundant_permuted_generators(self) -> None:
        result = compute_minimal_presentation(
            MinimalPresentationRequest(generators=("10", "9", "6", "4"))
        )

        assert result.minimal_generators == ("4", "6", "9")
        assert all(
            len(relation.first) == len(result.minimal_generators)
            and len(relation.second) == len(result.minimal_generators)
            for relation in result.relations
        )

    def test_presentation_result_rejects_a_redundant_coordinate_axis(self) -> None:
        with numerical_semigroup_error():
            MinimalPresentationResult.model_validate(
                {
                    "minimal_generators": ("3", "5", "8"),
                    "betti_elements": ("15",),
                    "relations": ({"first": [5, 0, 0], "second": [0, 3, 0]},),
                }
            )

    def test_presentation_3_5(self) -> None:
        req = MinimalPresentationRequest(generators=("3", "5"))
        result = compute_minimal_presentation(req)
        assert result.betti_elements == ("15",)
        assert len(result.relations) == 1
        assert result.relations[0].first == (5, 0)
        assert result.relations[0].second == (0, 3)

    def test_presentation_4_6_9(self) -> None:
        req = MinimalPresentationRequest(generators=("4", "6", "9"))
        result = compute_minimal_presentation(req)
        assert result.betti_elements == ("12", "18")
        assert len(result.relations) == 2

    def test_three_components_need_two_relations_not_all_pairs(self) -> None:
        result = compute_minimal_presentation(
            MinimalPresentationRequest(generators=("6", "10", "15"))
        )
        assert result.betti_elements == ("30",)
        assert len(result.relations) == 2
        for relation in result.relations:
            assert (
                sum(
                    coordinate * generator
                    for coordinate, generator in zip(
                        relation.first, (6, 10, 15), strict=True
                    )
                )
                == 30
            )
            assert (
                sum(
                    coordinate * generator
                    for coordinate, generator in zip(
                        relation.second, (6, 10, 15), strict=True
                    )
                )
                == 30
            )


class TestPresentationBinomials:
    def test_binomials_accept_relations_on_the_normalized_axis(self) -> None:
        presentation = compute_minimal_presentation(
            MinimalPresentationRequest(generators=("8", "5", "3"))
        )
        result = compute_presentation_binomials(
            PresentationBinomialsRequest(
                generators=("3", "8", "5"), relations=presentation.relations
            )
        )

        assert result.minimal_generators == ("3", "5")
        assert result.binomials[0].left_exponents == (5, 0)
        assert result.binomials[0].right_exponents == (0, 3)

    def test_binomial_result_rejects_a_redundant_coordinate_axis(self) -> None:
        with numerical_semigroup_error():
            PresentationBinomialsResult.model_validate(
                {
                    "minimal_generators": ("3", "5", "8"),
                    "binomials": (
                        {
                            "left_exponents": [5, 0, 0],
                            "right_exponents": [0, 3, 0],
                        },
                    ),
                }
            )

    def test_binomials_3_5(self) -> None:
        req = PresentationBinomialsRequest.model_validate(
            {
                "generators": ("3", "5"),
                "relations": [{"first": [5, 0], "second": [0, 3]}],
            }
        )
        result = compute_presentation_binomials(req)
        assert len(result.binomials) == 1
        b = result.binomials[0]
        assert b.left_coefficient == "1"
        assert b.left_exponents == (5, 0)
        assert b.right_coefficient == "-1"
        assert b.right_exponents == (0, 3)

    def test_binomials_4_6_9(self) -> None:
        req = PresentationBinomialsRequest.model_validate(
            {
                "generators": ("4", "6", "9"),
                "relations": [
                    {"first": [3, 0, 0], "second": [0, 2, 0]},
                    {"first": [0, 3, 0], "second": [0, 0, 2]},
                ],
            }
        )
        result = compute_presentation_binomials(req)
        assert len(result.binomials) == 2

    def test_binomials_rejects_empty_relations(self) -> None:
        result = compute_presentation_binomials(
            PresentationBinomialsRequest.model_validate(
                {"generators": ("1",), "relations": []}
            )
        )
        assert result.binomials == ()

    def test_binomials_reject_nonrelations(self) -> None:
        with operation_domain_error():
            compute_presentation_binomials(
                PresentationBinomialsRequest(
                    generators=("3", "5"),
                    relations=({"first": (1, 0), "second": (0, 1)},),
                )
            )


class TestGlobalDeltaSet:
    def test_delta_set_3_5(self) -> None:
        req = DeltaSetRequest(generators=("3", "5"))
        result = compute_delta_set(req)
        assert result.delta_set == (2,)

    def test_delta_set_4_6_9(self) -> None:
        req = DeltaSetRequest(generators=("4", "6", "9"))
        result = compute_delta_set(req)
        assert result.delta_set == (1,)

    def test_global_delta_is_not_only_union_of_betti_deltas(self) -> None:
        result = compute_delta_set(DeltaSetRequest(generators=("3", "8", "10")))
        assert result.delta_set == (1, 2, 3, 4)
        assert result.periodicity_bound == 96
        assert result.checked_through == 105


class TestGlobalElasticity:
    def test_elasticity_3_5(self) -> None:
        req = ElasticityRequest(generators=("3", "5"))
        result = compute_elasticity(req)
        assert result.elasticity == "5/3"

    def test_elasticity_4_6_9(self) -> None:
        req = ElasticityRequest(generators=("4", "6", "9"))
        result = compute_elasticity(req)
        assert result.elasticity == "9/4"

    def test_elasticity_2_3(self) -> None:
        req = ElasticityRequest(generators=("2", "3"))
        result = compute_elasticity(req)
        assert result.elasticity == "3/2"


class TestGlobalCatenaryDegree:
    def test_catenary_3_5(self) -> None:
        req = CatenaryDegreeRequest(generators=("3", "5"))
        result = compute_catenary_degree(req)
        assert result.catenary_degree == 5

    def test_catenary_4_6_9(self) -> None:
        req = CatenaryDegreeRequest(generators=("4", "6", "9"))
        result = compute_catenary_degree(req)
        assert result.catenary_degree == 3
        assert result.witness_betti_elements == ("12", "18")


class TestGeneratorEnvelopeIsSchemaVisible:
    """The per-generator ceiling is published wherever acceptance is advertised."""

    def test_request_schemas_state_the_per_generator_ceiling(self) -> None:
        for model in (
            BettiElementsRequest,
            CatenaryDegreeRequest,
            DeltaSetRequest,
            ElementCatenaryDegreeRequest,
            ElementDeltaSetRequest,
            ElementElasticityRequest,
            FactorizationComputeRequest,
            FactorizationDistanceRequest,
            FactorizationGraphComputeRequest,
            FactorizationLengthsComputeRequest,
            MinimalPresentationRequest,
            PresentationBinomialsRequest,
        ):
            schema = model.model_json_schema()
            description = schema["properties"]["generators"]["description"]
            assert f"each at most {MAX_GENERATOR}" in description
            assert "containing 1 canonicalizes to (1,)" in description

    def test_rejects_a_generator_above_the_published_ceiling(self) -> None:
        with operation_domain_error():
            compute_factorizations(
                FactorizationComputeRequest(generators=("2", "501"), value="503")
            )

    def test_free_axis_admits_a_redundant_generator_just_past_the_general_cap(
        self,
    ) -> None:
        result = compute_factorizations(
            FactorizationComputeRequest(
                generators=("1", str(MAX_GENERATOR + 1)),
                value=str(MAX_ELEMENT + 1),
            )
        )
        assert result.minimal_generators == ("1",)
        assert result.factorizations == ((MAX_ELEMENT + 1,),)

    def test_broadened_declarations_state_the_per_generator_ceiling(self) -> None:
        from jacobian.math.number_theory.numerical_semigroups._tools import TOOLS

        tools = {tool.operation_id: tool for tool in TOOLS}
        for operation_id in (
            "number_theory.numerical_semigroup.factorizations.compute",
            "number_theory.numerical_semigroup.presentation_binomials.compute",
        ):
            assert f"each at most {MAX_GENERATOR}" in tools[operation_id].description
        for operation_id in (
            "number_theory.numerical_semigroup.elasticity.compute",
            "number_theory.numerical_semigroup.elasticity.global_compute",
        ):
            examples = tools[operation_id].examples
            assert examples
            assert all(
                f"each at most {MAX_GENERATOR}" in example.description
                for example in examples
            )
