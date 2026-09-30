"""Contract tests for bounded braid and Wirtinger link operations."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    braid_closure,
    braid_inverse,
    braid_multiply,
    braid_permutation,
    link_alexander_polynomial,
    link_blackboard_graph,
    link_components,
    link_determinant,
    link_goeritz_data,
    link_linking_matrix,
    link_seifert_circles,
    wirtinger_presentation,
)
from jacobian.math.topology.links._extensions_models import (
    AlexanderPolynomialResult,
    BraidClosureResult,
    GoeritzDataResult,
    LinkBlackboardGraph,
    WirtingerPresentationResult,
)
from jacobian.math.topology.links._models import OrientedLinkDiagram


def _two_braid(*exponents: Literal[-1, 1]) -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=tuple(
            BraidLetter(generator=1, exponent=exponent) for exponent in exponents
        ),
    )


class TestBraidWords:
    def test_generator_axis_is_part_of_the_parent(self) -> None:
        with pytest.raises(ValidationError, match="generator index"):
            BraidWord(
                strand_count=2,
                letters=(BraidLetter(generator=2, exponent=1),),
            )

    def test_permutation_retains_closure_cycles_and_writhe(self) -> None:
        result = braid_permutation(_two_braid(1, 1, 1))

        assert result.permutation == (1, 0)
        assert result.cycles == ((0, 1),)
        assert result.closure_component_count == 1
        assert result.exponent_sum == 3

    def test_inverse_and_multiplication_preserve_the_braid_parent(self) -> None:
        word = _two_braid(1, -1, 1)
        inverse = braid_inverse(word)

        assert tuple(letter.exponent for letter in inverse.letters) == (-1, 1, -1)
        product = braid_multiply(word, inverse)
        assert product.strand_count == 2
        assert len(product.letters) == 6
        with pytest.raises(OperationDomainValidationError, match="strand count"):
            braid_multiply(word, BraidWord(strand_count=3))

    def test_braid_group_laws_match_independent_permutation_composition(self) -> None:
        left = BraidWord(
            strand_count=3,
            letters=(
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
            ),
        )
        right = BraidWord(
            strand_count=3,
            letters=(
                BraidLetter(generator=1, exponent=-1),
                BraidLetter(generator=1, exponent=1),
            ),
        )
        product = braid_multiply(left, right)
        inverse = braid_inverse(left)
        identity = braid_multiply(left, inverse)

        assert identity.strand_count == 3
        assert braid_permutation(identity).permutation == (0, 1, 2)
        assert braid_permutation(inverse).permutation == tuple(
            braid_permutation(left).permutation.index(i) for i in range(3)
        )
        p_left = braid_permutation(left).permutation
        p_right = braid_permutation(right).permutation
        assert braid_permutation(product).permutation == tuple(
            p_right[p_left[i]] for i in range(3)
        )
        assert braid_permutation(product).closure_component_count == len(
            braid_permutation(product).cycles
        )

    def test_empty_closure_retains_every_free_component(self) -> None:
        result = braid_closure(BraidWord(strand_count=3))

        assert result.diagram == OrientedLinkDiagram(free_loops=3)
        assert result.permutation.closure_component_count == 3
        assert len(link_components(result.diagram).components) == 3

    def test_two_positive_crossings_close_to_positive_hopf_link(self) -> None:
        result = braid_closure(_two_braid(1, 1))
        linking = link_linking_matrix(result.diagram)

        assert len(result.diagram.crossings) == 2
        assert result.permutation.closure_component_count == 2
        assert linking.matrix[0][1].as_fraction() == 1

    def test_closure_round_trips_with_source_transport(self) -> None:
        result = braid_closure(_two_braid(1, 1, 1))
        rebuilt = BraidClosureResult.model_validate_json(result.model_dump_json())

        assert rebuilt == result
        assert len(link_components(rebuilt.diagram).components) == 1

    def test_forged_braid_is_readmitted_before_execution(self) -> None:
        forged = BraidWord.model_construct(strand_count=0, letters=())

        with pytest.raises(OperationDomainValidationError, match="braid-word contract"):
            braid_permutation(forged)


class TestGoeritzData:
    def test_trefoil_goeritz_matrix_cross_checks_alexander_determinant(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = link_goeritz_data(link_blackboard_graph(diagram))

        assert result.reduced_matrix.entries == ((2, -1), (-1, 2))
        assert result.absolute_determinant == link_determinant(diagram).determinant == 3
        assert len(result.blackboard_graph.edges) == 3
        assert GoeritzDataResult.model_validate_json(result.model_dump_json()) == result

    def test_forged_tait_sign_is_rejected_before_a_wrong_determinant(self) -> None:
        """A caller-authored Tait sign must be recomputed, not trusted.

        The Goeritz matrix and the incidence numbers are built straight from it,
        so flipping one sign on a trefoil returned determinant 1 instead of 3.
        """
        graph = link_blackboard_graph(braid_closure(_two_braid(1, 1, 1)).diagram)
        edges = list(graph.edges)
        edges[0] = edges[0].model_copy(update={"tait_sign": -edges[0].tait_sign})
        forged = graph.model_copy(update={"edges": tuple(edges)})

        with pytest.raises(ValidationError) as error:
            LinkBlackboardGraph.model_validate(forged.model_dump())
        assert error.value.errors()[0]["type"] == "link_diagram.blackboard_tait_sign"

    def test_fabricated_region_incidence_is_rejected(self) -> None:
        """Region boundaries must be the diagram's own face cycles.

        Attributing a dart to the wrong region produced a Goeritz determinant of
        2 for a trefoil, and reversing a region's cyclic order was accepted too.
        """
        graph = link_blackboard_graph(braid_closure(_two_braid(1, 1, 1)).diagram)

        # swap one dart between two regions
        first, second = graph.regions[0], graph.regions[1]
        regions = [r.model_copy() for r in graph.regions]
        regions[0] = first.model_copy(
            update={
                "boundary_darts": first.boundary_darts[1:] + first.boundary_darts[:1]
            }
        )
        regions[1] = second.model_copy(
            update={"boundary_darts": second.boundary_darts + first.boundary_darts[:1]}
        )
        regions[0] = regions[0].model_copy(
            update={"boundary_darts": first.boundary_darts[1:]}
        )
        forged = graph.model_copy(update={"regions": tuple(regions)})
        # Depending on which dart moves, the first structural check to fire is
        # the edge-endpoint one or the face-cycle one; both are the point here,
        # since before the fix neither fired and the Goeritz matrix was built
        # from data the diagram does not support.
        with pytest.raises(ValidationError) as error:
            LinkBlackboardGraph.model_validate(forged.model_dump())
        assert error.value.errors()[0]["type"] in {
            "link_diagram.blackboard_edge_endpoints",
            "link_diagram.blackboard_region_faces",
        }

    def test_reversed_region_boundary_order_is_rejected(self) -> None:
        graph = link_blackboard_graph(braid_closure(_two_braid(1, 1, 1)).diagram)
        region = next(r for r in graph.regions if len(r.boundary_darts) > 1)
        regions = tuple(
            r.model_copy(update={"boundary_darts": tuple(reversed(r.boundary_darts))})
            if r.region_id == region.region_id
            else r
            for r in graph.regions
        )
        forged = graph.model_copy(update={"regions": regions})
        with pytest.raises(ValidationError) as error:
            LinkBlackboardGraph.model_validate(forged.model_dump())
        assert error.value.errors()[0]["type"] == "link_diagram.blackboard_region_faces"

    def test_permuted_face_axes_with_consistent_incidence_are_rejected(self) -> None:
        graph = link_blackboard_graph(braid_closure(_two_braid(1, 1, 1)).diagram)
        # Swap the seed face with an unshaded face and consistently recolor.
        # All cycles, corner incidences, and Tait signs remain individually valid;
        # only their attachment to the canonical region IDs is false.
        other = next(
            index for index, region in enumerate(graph.regions) if not region.shaded
        )
        permutation = list(range(len(graph.regions)))
        permutation[0], permutation[other] = permutation[other], permutation[0]
        regions = tuple(
            graph.regions[old_index].model_copy(
                update={
                    "region_id": f"region_{index:03d}",
                    "shaded": not graph.regions[old_index].shaded,
                }
            )
            for index, old_index in enumerate(permutation)
        )
        region_of = {
            dart: region for region in regions for dart in region.boundary_darts
        }
        edges = []
        for edge, crossing in zip(graph.edges, graph.diagram.crossings, strict=True):
            corners = tuple(
                index
                for index, dart in enumerate(crossing.half_edges)
                if region_of[dart].shaded
            )
            edges.append(
                edge.model_copy(
                    update={
                        "first_region_id": region_of[
                            crossing.half_edges[corners[0]]
                        ].region_id,
                        "second_region_id": region_of[
                            crossing.half_edges[corners[1]]
                        ].region_id,
                        "first_corner_index": corners[0],
                        "second_corner_index": corners[1],
                        "tait_sign": 1
                        if set(corners) == set(crossing.over_pair)
                        else -1,
                    }
                )
            )
        forged = graph.model_copy(
            update={
                "regions": regions,
                "shaded_region_ids": tuple(
                    region.region_id for region in regions if region.shaded
                ),
                "edges": tuple(edges),
            }
        )
        with pytest.raises(ValidationError) as error:
            LinkBlackboardGraph.model_validate_json(forged.model_dump_json())
        assert error.value.errors()[0]["type"] == "link_diagram.blackboard_region_faces"
        with pytest.raises(
            OperationDomainValidationError, match="bounded checkerboard graph contract"
        ):
            link_goeritz_data(forged)

    def test_mirror_negates_matrix_and_preserves_absolute_determinant(self) -> None:
        right = link_goeritz_data(
            link_blackboard_graph(braid_closure(_two_braid(1, 1, 1)).diagram)
        )
        left = link_goeritz_data(
            link_blackboard_graph(braid_closure(_two_braid(-1, -1, -1)).diagram)
        )

        assert left.reduced_matrix.entries == ((-2, 1), (1, -2))
        assert left.absolute_determinant == right.absolute_determinant

    def test_figure_eight_has_two_by_two_goeritz_matrix_of_determinant_five(
        self,
    ) -> None:
        word = BraidWord(
            strand_count=3,
            letters=(
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
            ),
        )
        diagram = braid_closure(word).diagram
        result = link_goeritz_data(link_blackboard_graph(diagram))

        assert result.reduced_matrix.entries == ((3, -2), (-2, 3))
        assert result.absolute_determinant == link_determinant(diagram).determinant == 5

    def test_goeritz_slice_rejects_crossing_free_and_over_bound_diagrams(self) -> None:
        with pytest.raises(OperationDomainValidationError, match="nonempty"):
            link_blackboard_graph(OrientedLinkDiagram(free_loops=1))

        over_bound = braid_closure(_two_braid(*(1,) * 33)).diagram
        with pytest.raises(OperationResourceAdmissionError, match="32 crossings"):
            link_goeritz_data(link_blackboard_graph(over_bound))


class TestSeifertCircles:
    def test_trefoil_surface_has_two_disks_three_bands_and_genus_one(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = link_seifert_circles(diagram)

        assert len(result.circles) == 2
        assert result.band_crossing_ids == tuple(
            crossing.crossing_id for crossing in diagram.crossings
        )
        assert result.euler_characteristic == -1
        assert result.genus == 1
        assert {dart for circle in result.circles for dart in circle.darts} == {
            dart for crossing in diagram.crossings for dart in crossing.half_edges
        }

    def test_crossing_free_unknot_is_one_seifert_disk(self) -> None:
        result = link_seifert_circles(OrientedLinkDiagram(free_loops=1))

        assert len(result.circles) == 1
        assert result.euler_characteristic == 1
        assert result.genus == 0

    def test_knot_first_contract_rejects_multiple_components(self) -> None:
        with pytest.raises(OperationDomainValidationError, match="one component"):
            link_seifert_circles(OrientedLinkDiagram(free_loops=2))


class TestAlexanderPolynomial:
    def test_unknot_has_unit_alexander_polynomial(self) -> None:
        result = link_alexander_polynomial(OrientedLinkDiagram(free_loops=1))

        assert [
            (term.coefficient.num, term.exponents[0])
            for term in result.polynomial.terms
        ] == [(1, 0)]

    def test_braid_trefoil_has_normalized_alexander_polynomial(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = link_alexander_polynomial(diagram)

        assert [
            (term.coefficient.num, term.exponents[0])
            for term in result.polynomial.terms
        ] == [(1, 2), (-1, 1), (1, 0)]
        assert (
            AlexanderPolynomialResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_figure_eight_has_independent_normalized_fixture(self) -> None:
        word = BraidWord(
            strand_count=3,
            letters=(
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
            ),
        )
        result = link_alexander_polynomial(braid_closure(word).diagram)

        assert [
            (term.coefficient.num, term.exponents[0])
            for term in result.polynomial.terms
        ] == [(1, 2), (-3, 1), (1, 0)]

    def test_trefoil_and_figure_eight_determinants_are_exact(self) -> None:
        trefoil = braid_closure(_two_braid(1, 1, 1)).diagram
        figure_eight_word = BraidWord(
            strand_count=3,
            letters=(
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=2, exponent=-1),
            ),
        )

        figure_eight = braid_closure(figure_eight_word).diagram

        for diagram, expected in ((trefoil, 3), (figure_eight, 5)):
            determinant = link_determinant(diagram)
            # Independent diagram route: the absolute reduced Goeritz determinant.
            goeritz = link_goeritz_data(link_blackboard_graph(diagram))
            assert determinant.determinant == expected
            assert determinant.determinant == goeritz.absolute_determinant
            assert determinant.alexander.diagram == diagram

    def test_one_variable_contract_rejects_links(self) -> None:
        hopf = braid_closure(_two_braid(1, 1)).diagram

        with pytest.raises(OperationDomainValidationError, match="one component"):
            link_alexander_polynomial(hopf)

    def test_determinant_expansion_has_an_explicit_crossing_bound(self) -> None:
        diagram = braid_closure(_two_braid(*(1,) * 9)).diagram

        with pytest.raises(OperationResourceAdmissionError, match="eight crossings"):
            link_alexander_polynomial(diagram)


class TestWirtingerPresentation:
    def test_zero_crossing_unknot_is_free_on_one_meridian(self) -> None:
        result = wirtinger_presentation(OrientedLinkDiagram(free_loops=1))

        assert result.presentation.generators == ("meridian_000",)
        assert result.presentation.relators == ()
        assert result.arcs[0].darts == ()

    def test_hopf_presentation_has_two_meridians_and_two_commutators(self) -> None:
        diagram = braid_closure(_two_braid(1, 1)).diagram
        result = wirtinger_presentation(diagram)

        assert len(result.arcs) == 2
        assert len(result.crossing_relators) == 2
        assert all(len(row.word.letters) == 4 for row in result.crossing_relators)
        assert tuple(row.word for row in result.crossing_relators) == (
            result.presentation.relators
        )

    def test_trefoil_presentation_keeps_crossing_and_dart_transport(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = wirtinger_presentation(diagram)

        assert len(result.arcs) == 3
        assert len(result.crossing_relators) == 3
        assert {dart for arc in result.arcs for dart in arc.darts} == {
            dart for crossing in diagram.crossings for dart in crossing.half_edges
        }
        assert (
            WirtingerPresentationResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_forged_diagram_is_readmitted_before_traversal(self) -> None:
        forged = OrientedLinkDiagram.model_construct(
            crossings=(), arcs=(), free_loops=-1
        )

        with pytest.raises(
            OperationDomainValidationError, match="oriented-link contract"
        ):
            wirtinger_presentation(forged)
