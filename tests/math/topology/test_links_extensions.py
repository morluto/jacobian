"""Contract tests for bounded braid and Wirtinger link operations."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    OrientedDiagramArc,
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
    link_mirror,
    link_seifert_circles,
    wirtinger_presentation,
)
from jacobian.math.topology.links._extensions_models import (
    AlexanderPolynomialRequest,
    AlexanderPolynomialResult,
    BlackboardGraphRequest,
    BraidClosureResult,
    BraidProductRequest,
    BraidWordRequest,
    GoeritzDataRequest,
    GoeritzDataResult,
    LinkBlackboardGraph,
    LinkDeterminantRequest,
    LinkDeterminantResult,
    SeifertCircleRequest,
    WirtingerPresentationRequest,
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

    def test_catalog_exposes_exact_group_word_operations(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        word = _two_braid(1, -1, 1)
        inverse = catalog["braid.word.inverse.compute"].run(BraidWordRequest(word=word))
        product = catalog["braid.word.multiply.compute"].run(
            BraidProductRequest(left=word, right=inverse)
        )

        assert product == BraidWord(
            strand_count=2, letters=word.letters + inverse.letters
        )
        assert braid_permutation(product).permutation == (0, 1)

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
    def test_hopf_tait_graph_is_two_signed_parallel_edges_and_round_trips(self) -> None:
        diagram = braid_closure(_two_braid(1, 1)).diagram
        graph = link_blackboard_graph(diagram)

        assert len(graph.shaded_region_ids) == 2
        assert len(graph.edges) == 2
        assert tuple(edge.crossing_id for edge in graph.edges) == tuple(
            crossing.crossing_id for crossing in diagram.crossings
        )
        assert all(edge.tait_sign == 1 for edge in graph.edges)
        assert all(
            edge.first_region_id != edge.second_region_id for edge in graph.edges
        )
        assert (
            len(
                {
                    frozenset((edge.first_region_id, edge.second_region_id))
                    for edge in graph.edges
                }
            )
            == 1
        )
        assert LinkBlackboardGraph.model_validate_json(graph.model_dump_json()) == graph

        forged = graph.model_dump(mode="json")
        forged["edges"][0]["tait_sign"] = -1
        with pytest.raises(ValidationError, match="Tait sign"):
            LinkBlackboardGraph.model_validate(forged)

        # The signed Tait graph Laplacian gives the Hopf Goeritz matrix [[2,-2],[-2,2]].
        laplacian = [[0, 0], [0, 0]]
        vertex = {
            region_id: index for index, region_id in enumerate(graph.shaded_region_ids)
        }
        for edge in graph.edges:
            left, right = vertex[edge.first_region_id], vertex[edge.second_region_id]
            laplacian[left][left] += edge.tait_sign
            laplacian[right][right] += edge.tait_sign
            laplacian[left][right] -= edge.tait_sign
            laplacian[right][left] -= edge.tait_sign
        assert laplacian == [[2, -2], [-2, 2]]
        goeritz = link_goeritz_data(diagram)
        assert goeritz.blackboard_graph == graph
        assert goeritz.reduced_matrix.entries == ((2,),)

        mirrored_graph = link_blackboard_graph(link_mirror(diagram).diagram)
        assert tuple(edge.tait_sign for edge in mirrored_graph.edges) == (-1, -1)

    def test_tait_graph_has_its_own_64_crossing_envelope(self) -> None:
        diagram = braid_closure(_two_braid(*(1,) * 33)).diagram

        graph = link_blackboard_graph(diagram)

        assert len(graph.edges) == 33
        assert len(graph.regions) == 35
        with pytest.raises(OperationResourceAdmissionError, match="32 crossings"):
            link_goeritz_data(diagram)

        boundary_diagram = braid_closure(_two_braid(*(1,) * 64)).diagram
        boundary_graph = link_blackboard_graph(boundary_diagram)
        assert len(boundary_graph.edges) == 64
        assert len(boundary_graph.regions) == 66

        # U+0001 expands to a six-byte JSON escape. Repeated maximum-length
        # labels exercise the output estimate's worst serialization case.
        def escaped_label(index: int) -> str:
            return "\x01" * 60 + f"{index:04}"

        dart_map = {
            dart: escaped_label(index)
            for index, dart in enumerate(
                dart
                for crossing in boundary_diagram.crossings
                for dart in crossing.half_edges
            )
        }
        escaped_diagram = OrientedLinkDiagram(
            crossings=tuple(
                crossing.model_copy(
                    update={
                        "crossing_id": escaped_label(1000 + index),
                        "half_edges": tuple(
                            dart_map[dart] for dart in crossing.half_edges
                        ),
                    }
                )
                for index, crossing in enumerate(boundary_diagram.crossings)
            ),
            arcs=tuple(
                OrientedDiagramArc(tail=dart_map[arc.tail], head=dart_map[arc.head])
                for arc in boundary_diagram.arcs
            ),
        )
        escaped_graph = link_blackboard_graph(escaped_diagram)
        assert len(escaped_graph.model_dump_json().encode()) <= 512 * 1024

    def test_catalog_publishes_reusable_tait_graph_value(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        diagram = braid_closure(_two_braid(1, 1)).diagram
        graph = catalog["link_diagram.blackboard_graph.compute"].run(
            BlackboardGraphRequest(diagram=diagram)
        )

        assert graph == link_blackboard_graph(diagram)

    def test_decoding_rejects_forged_face_cycle_and_color_seed(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        graph = link_blackboard_graph(diagram)

        malformed_face = graph.model_dump(mode="json")
        face_index = max(
            range(len(graph.regions)),
            key=lambda index: len(graph.regions[index].boundary_darts),
        )
        boundary = malformed_face["regions"][face_index]["boundary_darts"]
        assert len(boundary) > 2
        boundary.reverse()
        with pytest.raises(ValidationError, match="canonical face cycle"):
            LinkBlackboardGraph.model_validate_json(json.dumps(malformed_face))

        # Complement every color and consistently rebuild the edge incidence
        # data. This describes the other mathematical checkerboard graph, but
        # not the canonical color selected by the operation's least-face rule.
        swapped = graph.model_dump(mode="json")
        for region in swapped["regions"]:
            region["shaded"] = not region["shaded"]
        region_of_dart = {
            dart: region["region_id"]
            for region in swapped["regions"]
            for dart in region["boundary_darts"]
        }
        shaded = {
            region["region_id"] for region in swapped["regions"] if region["shaded"]
        }
        swapped["shaded_region_ids"] = [
            region["region_id"] for region in swapped["regions"] if region["shaded"]
        ]
        for edge, crossing in zip(swapped["edges"], diagram.crossings, strict=True):
            corners = tuple(
                index
                for index, dart in enumerate(crossing.half_edges)
                if region_of_dart[dart] in shaded
            )
            edge["first_corner_index"], edge["second_corner_index"] = corners
            edge["first_region_id"] = region_of_dart[crossing.half_edges[corners[0]]]
            edge["second_region_id"] = region_of_dart[crossing.half_edges[corners[1]]]
            edge["tait_sign"] = 1 if set(corners) == set(crossing.over_pair) else -1
        with pytest.raises(ValidationError, match="deterministic shaded color"):
            LinkBlackboardGraph.model_validate_json(json.dumps(swapped))

    def test_trefoil_goeritz_matrix_cross_checks_alexander_determinant(self) -> None:
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = link_goeritz_data(diagram)

        assert result.reduced_matrix.entries == ((2, -1), (-1, 2))
        assert result.absolute_determinant == link_determinant(diagram).determinant == 3
        assert len(result.blackboard_graph.edges) == 3
        assert GoeritzDataResult.model_validate_json(result.model_dump_json()) == result

    def test_mirror_negates_matrix_and_preserves_absolute_determinant(self) -> None:
        right = link_goeritz_data(braid_closure(_two_braid(1, 1, 1)).diagram)
        left = link_goeritz_data(braid_closure(_two_braid(-1, -1, -1)).diagram)

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
        result = link_goeritz_data(diagram)

        assert result.reduced_matrix.entries == ((3, -2), (-2, 3))
        assert result.absolute_determinant == link_determinant(diagram).determinant == 5

    def test_goeritz_slice_rejects_crossing_free_and_over_bound_diagrams(self) -> None:
        with pytest.raises(OperationDomainValidationError, match="nonempty"):
            link_goeritz_data(OrientedLinkDiagram(free_loops=1))

        over_bound = braid_closure(_two_braid(*(1,) * 33)).diagram
        with pytest.raises(OperationResourceAdmissionError, match="32 crossings"):
            link_goeritz_data(over_bound)


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
            goeritz = link_goeritz_data(diagram)
            assert determinant.determinant == expected
            assert determinant.determinant == goeritz.absolute_determinant
            assert determinant.alexander.diagram == diagram

    def test_determinant_is_a_public_source_bound_operation(self) -> None:
        tool = next(
            tool
            for tool in BUILTIN_TOOLS
            if tool.operation_id == "link_diagram.determinant.compute"
        )
        diagram = braid_closure(_two_braid(1, 1, 1)).diagram
        result = tool.run(LinkDeterminantRequest(diagram=diagram))

        assert isinstance(result, LinkDeterminantResult)
        assert result.alexander.diagram == diagram
        assert result.determinant == 3
        assert (
            LinkDeterminantResult.model_validate_json(result.model_dump_json())
            == result
        )

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


class TestLinkExtensionTools:
    def test_catalog_exposes_bounded_link_extension_operations(self) -> None:
        catalog = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
        assert (
            catalog["link_diagram.goeritz_matrix.compute"]
            .run(GoeritzDataRequest(diagram=braid_closure(_two_braid(1, 1)).diagram))
            .absolute_determinant
            == 2
        )
        assert (
            catalog["link_diagram.seifert_circles.compute"]
            .run(SeifertCircleRequest(diagram=OrientedLinkDiagram(free_loops=1)))
            .genus
            == 0
        )
        assert (
            catalog["link_diagram.alexander_polynomial.compute"]
            .run(AlexanderPolynomialRequest(diagram=OrientedLinkDiagram(free_loops=1)))
            .polynomial.terms[0]
            .coefficient.num
            == 1
        )
        assert (
            catalog["braid.word.permutation.compute"]
            .run(BraidWordRequest(word=_two_braid(1, 1, 1)))
            .closure_component_count
            == 1
        )
        assert (
            len(
                catalog["braid.word.closure.compute"]
                .run(BraidWordRequest(word=_two_braid(1, 1)))
                .diagram.crossings
            )
            == 2
        )
        assert catalog["link_diagram.wirtinger_presentation.compute"].run(
            WirtingerPresentationRequest(diagram=OrientedLinkDiagram(free_loops=1))
        ).presentation.generators == ("meridian_000",)
