"""Exact per-crossing sign and component-pair profiles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.topology.links import (
    LinkCrossing,
    LinkCrossingProfileResult,
    OrientedDiagramArc,
    OrientedLinkDiagram,
    link_crossing_profile,
)
from jacobian.math.topology.links._extensions_models import BraidLetter, BraidWord
from jacobian.math.topology.links.extensions import braid_closure
from jacobian.math.topology.links.extensions_tools import TOOLS
from jacobian.math.topology.links.operations import link_linking_matrix


def _hopf() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            LinkCrossing(
                crossing_id="c0",
                half_edges=("a0", "b0", "a1", "b1"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
            LinkCrossing(
                crossing_id="c1",
                half_edges=("a2", "b2", "a3", "b3"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
        ),
        arcs=(
            OrientedDiagramArc(tail="a1", head="b2"),
            OrientedDiagramArc(tail="a3", head="b0"),
            OrientedDiagramArc(tail="b1", head="a2"),
            OrientedDiagramArc(tail="b3", head="a0"),
        ),
    )


def _curl() -> OrientedLinkDiagram:
    return OrientedLinkDiagram(
        crossings=(
            LinkCrossing(
                crossing_id="c0",
                half_edges=("h0", "h1", "h2", "h3"),
                over_pair=(0, 2),
                under_pair=(1, 3),
                sign=-1,
            ),
        ),
        arcs=(
            OrientedDiagramArc(tail="h0", head="h3"),
            OrientedDiagramArc(tail="h1", head="h2"),
        ),
    )


def _component_roles_by_direct_traversal(
    diagram: OrientedLinkDiagram,
) -> dict[tuple[str, str], str]:
    """Independent small oracle: follow arc then opposite strand at each crossing."""
    arc_next = {arc.tail: arc.head for arc in diagram.arcs}
    strand_partner = {
        crossing.half_edges[left]: crossing.half_edges[right]
        for crossing in diagram.crossings
        for pair in (crossing.over_pair, crossing.under_pair)
        for left, right in (pair, pair[::-1])
    }
    crossing_roles = {
        crossing.half_edges[index]: (crossing.crossing_id, role)
        for crossing in diagram.crossings
        for pair, role in ((crossing.over_pair, "OVER"), (crossing.under_pair, "UNDER"))
        for index in pair
    }
    unseen = set(arc_next)
    role_component: dict[tuple[str, str], str] = {}
    component_index = 0
    while unseen:
        start = min(unseen)
        cursor = start
        component_darts: set[str] = set()
        while cursor not in component_darts:
            head = arc_next[cursor]
            component_darts.update((cursor, head))
            cursor = strand_partner[head]
        unseen.difference_update(component_darts)
        component_id = f"oracle_{component_index}"
        component_index += 1
        for dart in component_darts:
            if dart in crossing_roles:
                role_component[crossing_roles[dart]] = component_id
    return role_component


def test_crossing_profile_matches_independent_component_walk_and_writhe() -> None:
    for diagram, expected_signs in ((_hopf(), (-1, -1)), (_curl(), (-1,))):
        result = link_crossing_profile(diagram)
        oracle = _component_roles_by_direct_traversal(diagram)
        assert tuple(entry.sign for entry in result.crossings) == expected_signs
        assert result.writhe == sum(expected_signs)
        actual = {
            (entry.crossing_id, role): component_id
            for entry in result.crossings
            for role, component_id in (
                ("OVER", entry.over_component_id),
                ("UNDER", entry.under_component_id),
            )
        }
        role_keys = tuple(oracle)
        assert set(actual) == set(oracle)
        assert all(
            (oracle[left] == oracle[right]) == (actual[left] == actual[right])
            for left in role_keys
            for right in role_keys
        )


def test_crossing_profile_distinguishes_hopf_mixed_and_curl_self_crossings() -> None:
    hopf = link_crossing_profile(_hopf())
    assert all(
        entry.over_component_id != entry.under_component_id for entry in hopf.crossings
    )
    curl = link_crossing_profile(_curl())
    assert curl.crossings[0].over_component_id == curl.crossings[0].under_component_id


def test_positive_hopf_crossing_signs_match_standard_braid_and_linking_conventions() -> (
    None
):
    positive_hopf = braid_closure(
        BraidWord(
            strand_count=2,
            letters=(
                BraidLetter(generator=1, exponent=1),
                BraidLetter(generator=1, exponent=1),
            ),
        )
    ).diagram
    profile = link_crossing_profile(positive_hopf)
    linking = link_linking_matrix(positive_hopf)
    assert tuple(row.sign for row in profile.crossings) == (1, 1)
    assert profile.writhe == 2
    assert linking.matrix[0][1].as_fraction() == 1


def test_crossing_profile_handles_zero_crossing_unlink_and_serializes() -> None:
    result = link_crossing_profile(OrientedLinkDiagram(free_loops=3))
    assert result.crossings == ()
    assert result.writhe == 0
    assert (
        LinkCrossingProfileResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_crossing_profile_result_rejects_wrong_component_pair() -> None:
    result = link_crossing_profile(_hopf())
    payload = result.model_dump(mode="json")
    payload["crossings"][0]["over_component_id"] = "missing"
    with pytest.raises(ValidationError, match="over/under component IDs"):
        LinkCrossingProfileResult.model_validate(payload)


def test_crossing_profile_is_published_in_native_operation_manifest() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "link_diagram.crossing_profile.compute"
    )
    result = tool.run(tool.request_type.model_validate({"diagram": _curl()}))
    assert isinstance(result, LinkCrossingProfileResult)
    assert len(result.crossings) == 1
