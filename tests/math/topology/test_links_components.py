"""Tests for exact link-diagram component partitioning."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.topology.links import (
    ArcPairing,
    LinkCrossing,
    OrientedLinkDiagram,
)
from jacobian.math.topology.links._models import (
    LinkBracketRequest,
    LinkBracketResult,
    LinkComponentsRequest,
    LinkComponentsResult,
    LinkingMatrixResult,
    LinkJonesResult,
)
from jacobian.math.topology.links._tools import TOOLS, _run_bracket
from jacobian.math.topology.links.operations import (
    link_bracket,
    link_components,
    link_jones,
    link_linking_matrix,
)


def _crossing(
    cid: str, darts: tuple[str, str, str, str], over: tuple[int, int] = (0, 2)
) -> LinkCrossing:
    under = (1, 3) if over == (0, 2) else (0, 2)
    return LinkCrossing(
        crossing_id=cid, half_edges=darts, over_pair=over, under_pair=under
    )


def _arc(first: str, second: str) -> ArcPairing:
    return ArcPairing(first=first, second=second)


def _hopf() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            _crossing("c0", ("a0", "b0", "a1", "b1")),
            _crossing("c1", ("a2", "b2", "a3", "b3")),
        ),
        arcs=(
            _arc("a1", "b2"),
            _arc("a3", "b0"),
            _arc("b1", "a2"),
            _arc("b3", "a0"),
        ),
    )


def _unknot_curl() -> OrientedLinkDiagram:
    # One Reidemeister-I curl: a single component visiting one crossing twice
    # (once over, once under).
    return OrientedLinkDiagram(
        crossings=(_crossing("c0", ("h0", "h1", "h2", "h3")),),
        arcs=(_arc("h0", "h3"), _arc("h1", "h2")),
    )


class TestKnownAnswer:
    def test_free_loop_multiplies_a_crossing_bracket(self) -> None:
        crossing = _unknot_curl()
        without_loop = link_bracket(crossing)
        with_loop = link_bracket(crossing.model_copy(update={"free_loops": 1}))

        # A disjoint unknot contributes delta = -A^2 - A^-2 to every state.
        expected = {-1: 1, -5: 1}
        assert {
            term.exponents[0]: term.coefficient.as_fraction()
            for term in with_loop.polynomial.terms
        } == expected
        assert {
            term.exponents[0]: term.coefficient.as_fraction()
            for term in without_loop.polynomial.terms
        } == {-3: -1}
        assert all(
            state.circle_count == base.circle_count + 1
            for state, base in zip(with_loop.states, without_loop.states, strict=True)
        )

    def test_serialized_result_axes_are_source_bound(self) -> None:
        bracket = link_bracket(_unknot_curl())
        payload = bracket.model_dump(mode="json")
        payload["states"][0]["choices"] = []
        with pytest.raises(ValidationError):
            LinkBracketResult.model_validate(payload)
        payload = bracket.model_dump(mode="json")
        payload["states"][1]["choices"] = payload["states"][0]["choices"]
        with pytest.raises(ValidationError):
            LinkBracketResult.model_validate(payload)

        jones = link_jones(_unknot_curl())
        payload = jones.model_dump(mode="json")
        payload["diagram"] = {"free_loops": 1}
        with pytest.raises(ValidationError):
            LinkJonesResult.model_validate(payload)

        linking = link_linking_matrix(_hopf())
        payload = linking.model_dump(mode="json")
        payload["component_ids"] = ["wrong", "axis"]
        with pytest.raises(ValidationError):
            LinkingMatrixResult.model_validate(payload)

    def test_serialized_exact_results_reject_forged_mathematics(self) -> None:
        bracket = link_bracket(_unknot_curl())
        payload = bracket.model_dump(mode="json")
        payload["states"][0]["circle_count"] += 1
        with pytest.raises(ValidationError):
            LinkBracketResult.model_validate(payload)
        payload = bracket.model_dump(mode="json")
        payload["polynomial"]["terms"] = []
        with pytest.raises(ValidationError):
            LinkBracketResult.model_validate(payload)

        jones = link_jones(_unknot_curl())
        payload = jones.model_dump(mode="json")
        payload["polynomial"]["terms"] = []
        with pytest.raises(ValidationError):
            LinkJonesResult.model_validate(payload)

        linking = link_linking_matrix(_hopf())
        payload = linking.model_dump(mode="json")
        forged_value = {"num": "7", "den": "1"}
        payload["matrix"][0][1] = forged_value
        payload["matrix"][1][0] = forged_value
        with pytest.raises(ValidationError):
            LinkingMatrixResult.model_validate(payload)

    def test_serialized_components_reject_noncycle_partitions(self) -> None:
        result = link_components(_hopf())
        payload = result.model_dump(mode="json")
        first = payload["components"][0]["darts"]
        second = payload["components"][1]["darts"]
        first[0], second[0] = second[0], first[0]
        dart_roles = {
            dart: {
                "crossing_id": crossing.crossing_id,
                "role": (
                    "OVER"
                    if dart in {crossing.half_edges[i] for i in crossing.over_pair}
                    else "UNDER"
                ),
            }
            for crossing in result.diagram.crossings
            for dart in crossing.half_edges
        }
        for component in payload["components"]:
            component["visits"] = sorted(
                (dart_roles[dart] for dart in component["darts"]),
                key=lambda visit: visit["crossing_id"],
            )
        with pytest.raises(ValidationError):
            LinkComponentsResult.model_validate(payload)

    def test_serialized_components_retain_source_diagram(self) -> None:
        result = link_components(_hopf())
        payload = result.model_dump(mode="json")
        payload["diagram"] = {"free_loops": 1}
        with pytest.raises(ValidationError):
            LinkComponentsResult.model_validate(payload)

    def test_free_loop_also_multiplies_writhe_normalized_jones(self) -> None:
        crossing = _unknot_curl()
        without_loop = link_jones(crossing)
        with_loop = link_jones(crossing.model_copy(update={"free_loops": 1}))

        expected: dict[int, Fraction] = {}
        for term in without_loop.polynomial.terms:
            coefficient = -term.coefficient.as_fraction()
            expected[term.exponents[0] + 2] = coefficient
            expected[term.exponents[0] - 2] = coefficient
        actual = {
            term.exponents[0]: term.coefficient.as_fraction()
            for term in with_loop.polynomial.terms
        }
        assert actual == expected
        assert (
            with_loop.bracket.polynomial
            == link_bracket(crossing.model_copy(update={"free_loops": 1})).polynomial
        )

    def test_disjoint_union_regression_through_catalog_wrapper(self) -> None:
        diagram = _unknot_curl().model_copy(update={"free_loops": 1})
        request = LinkBracketRequest.model_validate_json(
            LinkBracketRequest(diagram=diagram).model_dump_json()
        )
        next(
            tool
            for tool in TOOLS
            if tool.operation_id == "link_diagram.bracket.compute"
        )
        assert _run_bracket(request) == link_bracket(diagram)

    def test_hopf_link_has_two_components(self) -> None:
        result = link_components(_hopf())
        assert result.component_count == 2
        assert len(result.components) == 2
        for component in result.components:
            assert component.length == 4
            roles = {visit.role for visit in component.visits}
            assert roles == {"OVER", "UNDER"} or roles == {"OVER"} or roles == {"UNDER"}

    def test_hopf_component_roles(self) -> None:
        result = link_components(_hopf())
        visits = {
            component.component_id: {
                visit.crossing_id: visit.role for visit in component.visits
            }
            for component in result.components
        }
        assert visits["component_000"] == {"c0": "OVER", "c1": "UNDER"}
        assert visits["component_001"] == {"c0": "UNDER", "c1": "OVER"}

    def test_curl_is_one_component(self) -> None:
        result = link_components(_unknot_curl())
        assert result.component_count == 1
        assert result.components[0].length == 4


class TestBoundary:
    def test_free_loop_component(self) -> None:
        diagram = OrientedLinkDiagram(free_loops=1)
        result = link_components(diagram)
        assert result.component_count == 1
        assert result.components[0].visits == ()

    def test_two_free_loops_unlink(self) -> None:
        diagram = OrientedLinkDiagram(free_loops=2)
        result = link_components(diagram)
        assert result.component_count == 2

    def test_empty_diagram_rejected_at_value_boundary(self) -> None:
        with pytest.raises(ValidationError):
            OrientedLinkDiagram()


class TestAdversarial:
    def test_repeated_dart_rejected_at_value_boundary(self) -> None:
        with pytest.raises(ValidationError):
            OrientedLinkDiagram(
                crossings=(_crossing("c0", ("h0", "h1", "h0", "h3")),),
                arcs=(_arc("h0", "h1"), _arc("h2", "h3")),
            )

    def test_unpaired_dart_rejected_at_value_boundary(self) -> None:
        with pytest.raises(ValidationError):
            OrientedLinkDiagram(
                crossings=(_crossing("c0", ("h0", "h1", "h2", "h3")),),
                arcs=(_arc("h0", "h1"),),
            )

    def test_native_rejects_non_diagram(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            link_components("not-a-diagram")  # type: ignore[arg-type]


class TestDefiningInvariant:
    def test_every_dart_covered_exactly_once(self) -> None:
        diagram = _hopf()
        result = link_components(diagram)
        covered = [dart for component in result.components for dart in component.darts]
        expected = [
            dart for crossing in diagram.crossings for dart in crossing.half_edges
        ]
        assert sorted(covered) == sorted(expected)
        assert len(covered) == len(set(covered)) == 8

    def test_disjoint_cycles_partition_arcs(self) -> None:
        result = link_components(_hopf())
        seen: set[str] = set()
        for component in result.components:
            for dart in component.darts:
                assert dart not in seen
                seen.add(dart)
        assert len(seen) == 8

    def test_curl_visits_both_roles(self) -> None:
        result = link_components(_unknot_curl())
        roles = {visit.role for visit in result.components[0].visits}
        assert roles == {"OVER", "UNDER"}


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = LinkComponentsRequest(diagram=_hopf())
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "link_diagram.components.compute"
        )
        component_tool = cast(
            MathTool[LinkComponentsRequest, LinkComponentsResult], tool
        )
        assert component_tool.run(request) == link_components(_hopf())

    def test_operation_is_published(self) -> None:
        assert "link_diagram.components.compute" in {
            tool.operation_id for tool in TOOLS
        }
