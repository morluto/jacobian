"""Independent rotation/orientation checks for the V2 diagram source value."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.topology.links._extensions_models import BraidLetter, BraidWord
from jacobian.math.topology.links._models import (
    LinkCrossing,
    OrientedDiagramArc,
    OrientedLinkDiagram,
)
from jacobian.math.topology.links.extensions import (
    braid_closure,
    link_alexander_polynomial,
)
from jacobian.math.topology.links.operations import link_jones, link_linking_matrix


def _oriented_closure(word: BraidWord) -> OrientedLinkDiagram:
    """The typed braid constructor provides genuine component orientation."""
    return braid_closure(word).diagram


def _word(strands: int, letters: tuple[int, ...]) -> BraidWord:
    return BraidWord(
        strand_count=strands,
        letters=tuple(
            BraidLetter(generator=abs(i), exponent=1 if i > 0 else -1) for i in letters
        ),
    )


def _components(diagram: OrientedLinkDiagram) -> int:
    partner = {}
    for crossing in diagram.crossings:
        for pair in (crossing.over_pair, crossing.under_pair):
            a, b = (crossing.half_edges[i] for i in pair)
            partner[a] = b
            partner[b] = a
    outgoing = {arc.tail: arc.head for arc in diagram.arcs}
    unseen = set(outgoing)
    count = diagram.free_loops
    while unseen:
        count += 1
        dart = min(unseen)
        while dart in unseen:
            unseen.remove(dart)
            dart = partner[outgoing[dart]]
    return count


def test_braid_closures_encode_oriented_planar_trefoil_hopf_and_figure_eight() -> None:
    trefoil = _oriented_closure(_word(2, (1, 1, 1)))
    hopf = _oriented_closure(_word(2, (1, 1)))
    figure_eight = _oriented_closure(_word(3, (1, -2, 1, -2)))

    assert _components(trefoil) == 1
    assert tuple(c.sign for c in trefoil.crossings) == (1, 1, 1)
    assert _components(hopf) == 2
    assert tuple(c.sign for c in hopf.crossings) == (1, 1)
    assert _components(figure_eight) == 1
    assert tuple(c.sign for c in figure_eight.crossings) == (1, -1, 1, -1)

    trefoil_alexander = link_alexander_polynomial(trefoil).polynomial
    figure_eight_alexander = link_alexander_polynomial(figure_eight).polynomial
    assert [
        (term.coefficient.as_fraction(), term.exponents[0])
        for term in trefoil_alexander.terms
    ] == [(1, 2), (-1, 1), (1, 0)]
    assert [
        (term.coefficient.as_fraction(), term.exponents[0])
        for term in figure_eight_alexander.terms
    ] == [(1, 2), (-3, 1), (1, 0)]
    hopf_linking = link_linking_matrix(hopf).matrix
    assert hopf_linking[0][1].as_fraction() == 1


def test_orientation_and_rotation_fail_closed_for_legacy_or_forged_values() -> None:
    legacy_wire = {
        "crossings": [
            {
                "crossing_id": "c",
                "half_edges": ["h0", "h1", "h2", "h3"],
                "over_pair": [0, 2],
                "under_pair": [1, 3],
                "sign": 1,
            }
        ],
        "arcs": [{"first": "h0", "second": "h1"}, {"first": "h2", "second": "h3"}],
    }
    with pytest.raises(ValidationError):
        OrientedLinkDiagram.model_validate(legacy_wire)

    diagram = _oriented_closure(_word(2, (1, 1, 1)))
    forged = diagram.model_dump()
    forged["crossings"][0]["sign"] = -1
    with pytest.raises(ValidationError, match="sign disagrees"):
        OrientedLinkDiagram.model_validate(forged)


def test_torus_rotation_system_is_rejected_even_when_pairings_are_complete() -> None:
    crossing = LinkCrossing(
        crossing_id="c",
        half_edges=("h0", "h1", "h2", "h3"),
        over_pair=(0, 2),
        under_pair=(1, 3),
        sign=1,
    )
    # Pairing opposite darts produces a one-vertex genus-one rotation system.
    crossing = crossing.model_copy(update={"sign": -1})
    with pytest.raises(ValidationError, match="not an embedding on the sphere"):
        OrientedLinkDiagram(
            crossings=(crossing,),
            arcs=(
                OrientedDiagramArc(tail="h0", head="h2"),
                OrientedDiagramArc(tail="h1", head="h3"),
            ),
            free_loops=0,
        )


def _legacy_projection(value: OrientedLinkDiagram) -> OrientedLinkDiagram:
    """Explicit one-way projection after V2 orientation has been validated."""
    return value


def test_mirror_reverses_checked_crossing_signs_and_preserves_sphere_embedding() -> (
    None
):
    source = _oriented_closure(_word(2, (1, 1, 1)))
    crossings = tuple(
        crossing.model_copy(
            update={
                "over_pair": crossing.under_pair,
                "under_pair": crossing.over_pair,
                "sign": -crossing.sign,
            }
        )
        for crossing in source.crossings
    )
    mirrored = OrientedLinkDiagram(
        crossings=crossings, arcs=source.arcs, free_loops=source.free_loops
    )
    assert tuple(c.sign for c in mirrored.crossings) == (-1, -1, -1)
    assert _components(mirrored) == _components(source)
    assert tuple(c.sign for c in mirrored.crossings) == tuple(
        -c.sign for c in source.crossings
    )

    jones = link_jones(source).polynomial
    mirror_jones = link_jones(mirrored).polynomial
    assert {
        term.exponents[0]: term.coefficient.as_fraction() for term in mirror_jones.terms
    } == {-term.exponents[0]: term.coefficient.as_fraction() for term in jones.terms}

    # Repeat the independent Jones mirror relation on the two-component Hopf
    # link and the figure-eight knot, not just the trefoil fixture.
    for fixture in (
        _oriented_closure(_word(2, (1, 1))),
        _oriented_closure(_word(3, (1, -2, 1, -2))),
    ):
        fixture_mirror = OrientedLinkDiagram(
            crossings=tuple(
                crossing.model_copy(
                    update={
                        "over_pair": crossing.under_pair,
                        "under_pair": crossing.over_pair,
                        "sign": -crossing.sign,
                    }
                )
                for crossing in fixture.crossings
            ),
            arcs=fixture.arcs,
            free_loops=fixture.free_loops,
        )
        original_terms = link_jones(fixture).polynomial.terms
        mirror_terms = link_jones(fixture_mirror).polynomial.terms
        if len(fixture.crossings) == 2:
            assert link_linking_matrix(fixture_mirror).matrix[0][1].as_fraction() == -1
        assert {
            term.exponents[0]: term.coefficient.as_fraction() for term in mirror_terms
        } == {
            -term.exponents[0]: term.coefficient.as_fraction()
            for term in original_terms
        }


def test_diagram_bound_accepts_64_crossings_and_rejects_65_before_validation() -> None:
    crossings = tuple(
        LinkCrossing(
            crossing_id=f"c{index:03d}",
            half_edges=tuple(f"c{index:03d}:h{dart}" for dart in range(4)),
            over_pair=(0, 2),
            under_pair=(1, 3),
            sign=-1,
        )
        for index in range(64)
    )
    arcs = tuple(
        arc
        for crossing in crossings
        for arc in (
            OrientedDiagramArc(
                tail=crossing.half_edges[0], head=crossing.half_edges[3]
            ),
            OrientedDiagramArc(
                tail=crossing.half_edges[1], head=crossing.half_edges[2]
            ),
        )
    )
    assert len(OrientedLinkDiagram(crossings=crossings, arcs=arcs).crossings) == 64
    with pytest.raises(ValidationError, match="at most 64"):
        OrientedLinkDiagram(crossings=(*crossings, crossings[-1]), arcs=arcs)
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        OrientedLinkDiagram(crossings=(), arcs=(), free_loops=65)


def test_reidemeister_and_braid_relation_closures_preserve_normalized_jones() -> None:
    curl = braid_closure(_word(2, (1,))).diagram
    unknot = braid_closure(_word(1, ())).diagram
    assert link_jones(curl).polynomial.terms == link_jones(unknot).polynomial.terms

    r2_pair = braid_closure(_word(2, (1, -1))).diagram
    two_component_unlink = braid_closure(_word(2, ())).diagram
    assert (
        link_jones(r2_pair).polynomial.terms
        == link_jones(two_component_unlink).polynomial.terms
    )

    relation_left = braid_closure(_word(3, (1, 2, 1))).diagram
    relation_right = braid_closure(_word(3, (2, 1, 2))).diagram
    assert (
        link_jones(relation_left).polynomial.terms
        == link_jones(relation_right).polynomial.terms
    )


def test_reversing_one_hopf_component_changes_only_its_mixed_crossing_signs() -> None:
    source = _oriented_closure(_word(2, (1, 1)))
    selected_component_arcs = {source.arcs[0], source.arcs[3]}
    reversed_arcs = tuple(
        OrientedDiagramArc(tail=arc.head, head=arc.tail)
        if arc in selected_component_arcs
        else arc
        for arc in source.arcs
    )
    # Reversing one braid-closure component reverses its two directed arcs;
    # both crossings are mixed, so both crossing signs change.
    reversed_component = OrientedLinkDiagram(
        crossings=tuple(
            crossing.model_copy(update={"sign": -crossing.sign})
            for crossing in source.crossings
        ),
        arcs=reversed_arcs,
        free_loops=source.free_loops,
    )
    assert _components(reversed_component) == 2
    assert tuple(c.sign for c in reversed_component.crossings) == (-1, -1)
    assert link_linking_matrix(reversed_component).matrix[0][1].as_fraction() == -1
