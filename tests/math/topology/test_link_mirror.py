"""Exact mirror transform for oriented classical link diagrams."""

from fractions import Fraction

from jacobian.math.topology.links import (
    LinkCrossing,
    OrientedDiagramArc,
    OrientedLinkDiagram,
    link_bracket,
    link_jones,
    link_mirror,
)
from jacobian.math.topology.links._models import LinkDiagramMirrorResult
from jacobian.math.topology.links._tools import TOOLS


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


def _terms(poly: object) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in poly.terms  # type: ignore[attr-defined]
    }


def test_mirror_swaps_crossing_roles_and_is_an_involution() -> None:
    diagram = _curl()
    mirrored = link_mirror(diagram)

    assert mirrored.source == diagram
    assert mirrored.diagram.crossings[0].over_pair == (1, 3)
    assert mirrored.diagram.crossings[0].under_pair == (0, 2)
    assert mirrored.diagram.crossings[0].sign == 1
    assert mirrored.diagram.arcs == diagram.arcs
    assert link_mirror(mirrored.diagram).diagram == diagram


def test_mirror_replays_bracket_and_jones_variable_inversion() -> None:
    diagram = _curl()
    mirror = link_mirror(diagram).diagram

    assert _terms(link_bracket(mirror).polynomial) == {
        -power: value
        for power, value in _terms(link_bracket(diagram).polynomial).items()
    }
    assert _terms(link_jones(mirror).polynomial) == {
        -power: value for power, value in _terms(link_jones(diagram).polynomial).items()
    }


def test_result_serialization_retains_identity_transport_contract() -> None:
    result = link_mirror(_curl())
    assert (
        LinkDiagramMirrorResult.model_validate_json(result.model_dump_json()) == result
    )


def test_catalog_declares_mirror_transform() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "link_diagram.mirror.compute"
    )
    assert (
        tool.run(
            tool.request_type.model_validate(
                {"diagram": _curl().model_dump(mode="json")}
            )
        ).diagram
        == link_mirror(_curl()).diagram
    )
