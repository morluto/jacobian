"""Exact source-bound reversal of selected link component orientations."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.links import (
    LinkCrossing,
    OrientedDiagramArc,
    OrientedLinkDiagram,
    link_bracket,
    link_components,
    link_linking_matrix,
    link_orientation_reverse,
)
from jacobian.math.topology.links._tools import TOOLS


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


def test_reverse_one_hopf_component_flips_mixed_signs_and_linking() -> None:
    source = _hopf()
    source_components = link_components(source)
    selected = source_components.components[0]
    result = link_orientation_reverse(source, (selected.darts[0],))

    assert result.source == source
    assert result.diagram.arcs == tuple(
        OrientedDiagramArc(tail=arc.head, head=arc.tail)
        if arc.tail in selected.darts
        else arc
        for arc in source.arcs
    )
    assert [crossing.sign for crossing in result.diagram.crossings] == [1, 1]
    assert {change.crossing_id for change in result.crossing_sign_changes} == {
        "c0",
        "c1",
    }
    assert len(result.component_transport) == 2
    assert all(len(row.darts) == 4 for row in result.component_transport)
    before = link_linking_matrix(source).matrix
    after = link_linking_matrix(result.diagram).matrix
    assert before[0][1].as_fraction() == Fraction(-1)
    assert after[0][1].as_fraction() == Fraction(1)
    assert link_bracket(result.diagram).polynomial == link_bracket(source).polynomial
    restored = link_orientation_reverse(result.diagram, (selected.darts[0],))
    assert restored.diagram == source


def test_reverse_both_hopf_components_preserves_crossing_signs() -> None:
    source = _hopf()
    representatives = tuple(
        component.darts[0] for component in link_components(source).components
    )
    result = link_orientation_reverse(source, representatives)
    assert [crossing.sign for crossing in result.diagram.crossings] == [-1, -1]
    assert result.crossing_sign_changes == ()
    assert all(row.orientation_reversed for row in result.component_transport)
    assert (
        link_linking_matrix(result.diagram).matrix == link_linking_matrix(source).matrix
    )


def test_reversal_noop_and_free_loop_representative_rejection() -> None:
    source = OrientedLinkDiagram(
        crossings=_hopf().crossings, arcs=_hopf().arcs, free_loops=1
    )
    assert link_orientation_reverse(source).diagram == source
    with pytest.raises(
        OperationDomainValidationError, match="free loops are untracked"
    ):
        link_orientation_reverse(source, ("free_loop_000:dart",))


def test_orientation_reversal_catalog_example_executes() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "link_diagram.orientation_reverse.compute"
    )
    example = tool.examples[0]
    result = tool.run(tool.request_type.model_validate(example.input))
    assert len(result.crossing_sign_changes) == 2
