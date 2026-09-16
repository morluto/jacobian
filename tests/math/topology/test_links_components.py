"""Tests for exact link-diagram component partitioning."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.links import (
    ArcPairing,
    LinkCrossing,
    OrientedLinkDiagram,
)
from jacobian.math.topology.links._models import LinkComponentsRequest
from jacobian.math.topology.links._tools import TOOLS
from jacobian.math.topology.links.operations import link_components


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
        assert tool.run(request) == link_components(_hopf())

    def test_operation_is_published(self) -> None:
        assert "link_diagram.components.compute" in {
            tool.operation_id for tool in TOOLS
        }
