"""Owning tests for the source-transporting link disjoint union.

The published example unions two crossing-free unknots, so the crossing,
dart, and arc transport of a genuine pair of crossing-bearing links was
exercised nowhere. These tests use a trefoil and a Hopf link.
"""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.links import (
    BraidLetter,
    BraidWord,
    braid_closure,
    link_components,
    link_crossing_profile,
    link_disjoint_union,
)
from jacobian.math.topology.links._extensions_models import (
    LinkDisjointUnionResult,
)
from jacobian.math.topology.links._models import MAX_LINK_CROSSINGS


def _trefoil() -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=tuple(BraidLetter(generator=1, exponent=1) for _ in range(3)),
    )


def _hopf() -> BraidWord:
    return BraidWord(
        strand_count=2,
        letters=(
            BraidLetter(generator=1, exponent=1),
            BraidLetter(generator=1, exponent=-1),
        ),
    )


def _union() -> LinkDisjointUnionResult:
    return link_disjoint_union(
        (braid_closure(_trefoil()).diagram, braid_closure(_hopf()).diagram)
    )


def test_disjoint_union_transports_every_crossing_dart_and_arc() -> None:
    trefoil = braid_closure(_trefoil()).diagram
    hopf = braid_closure(_hopf()).diagram
    result = _union()

    assert len(trefoil.crossings) == 3
    assert len(hopf.crossings) == 2
    assert len(result.sources) == 2
    assert result.sources == (trefoil, hopf)
    assert len(result.diagram.crossings) == 5
    assert len(result.diagram.arcs) == len(trefoil.arcs) + len(hopf.arcs)
    assert result.diagram.free_loops == 0

    assert len(result.crossing_map) == 5
    assert len(result.dart_map) == 4 * 5
    assert len(result.arc_map) == len(result.diagram.arcs)
    assert len({row.target_crossing_id for row in result.crossing_map}) == 5
    assert len({row.target_dart_id for row in result.dart_map}) == 4 * 5
    assert {row.source_index for row in result.crossing_map} == {0, 1}


def test_disjoint_union_keeps_each_source_component_distinct() -> None:
    trefoil = braid_closure(_trefoil()).diagram
    hopf = braid_closure(_hopf()).diagram
    result = _union()

    union_components = link_components(result.diagram)
    assert union_components.component_count == 3
    assert link_components(trefoil).component_count == 1
    assert link_components(hopf).component_count == 2

    # Independent invariants of the two sources, recomputed on the union's
    # transport rather than read back from the retained sources.
    trefoil_index = {
        row.source_crossing_id: row.target_crossing_id
        for row in result.crossing_map
        if row.source_index == 0
    }
    assert len(trefoil_index) == 3
    profile = link_crossing_profile(result.diagram)
    assert len(profile.crossings) == 5
    assert profile.components.component_count == 3

    # Every union component must stay inside one source's transported darts,
    # which is what "disjoint" establishes.
    dart_owner = {row.target_dart_id: row.source_index for row in result.dart_map}
    for component in union_components.components:
        assert len({dart_owner[dart] for dart in component.darts}) == 1
    assert {
        dart_owner[component.darts[0]] for component in union_components.components
    } == {
        0,
        1,
    }


def test_disjoint_union_result_round_trips_through_its_carrier() -> None:
    result = _union()

    revived = LinkDisjointUnionResult.model_validate(result.model_dump())

    assert revived == result
    assert revived.diagram.crossings == result.diagram.crossings
    assert revived.arc_map == result.arc_map
    assert revived.free_loop_map == result.free_loop_map


def test_disjoint_union_of_one_diagram_is_its_tagged_copy() -> None:
    trefoil = braid_closure(_trefoil()).diagram

    result = link_disjoint_union((trefoil,))

    assert result.sources == (trefoil,)
    assert len(result.diagram.crossings) == len(trefoil.crossings)
    assert {row.target_crossing_id for row in result.crossing_map} == {
        f"link_00_crossing_{index:03d}" for index in range(len(trefoil.crossings))
    }


def test_disjoint_union_offsets_free_loop_indices_across_sources() -> None:
    import json

    first = braid_closure(BraidWord(strand_count=2)).diagram.model_copy(
        update={"free_loops": 2}
    )
    second = braid_closure(BraidWord(strand_count=2)).diagram.model_copy(
        update={"free_loops": 3}
    )

    result = link_disjoint_union((first, second))

    assert result.diagram.free_loops == 5
    assert [
        (row.source_index, row.target_loop_index) for row in result.free_loop_map
    ] == [
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]
    assert json.loads(result.model_dump_json())["diagram"]["free_loops"] == 5


@pytest.mark.parametrize("arity", (0, MAX_LINK_CROSSINGS + 1))
def test_disjoint_union_rejects_an_unsupported_arity(arity: int) -> None:
    unknot = braid_closure(BraidWord(strand_count=2)).diagram
    family = (unknot,) * arity

    with pytest.raises(OperationDomainValidationError) as error:
        link_disjoint_union(family)

    assert error.value.errors()[0]["type"] == "link_diagram.disjoint_union_arity"


def test_disjoint_union_rejects_an_aggregate_crossing_overflow() -> None:
    half = BraidWord(
        strand_count=2,
        letters=tuple(
            BraidLetter(generator=1, exponent=1)
            for _ in range(MAX_LINK_CROSSINGS // 2 + 1)
        ),
    )
    diagram = braid_closure(half).diagram
    assert len(diagram.crossings) == MAX_LINK_CROSSINGS // 2 + 1

    with pytest.raises(OperationResourceAdmissionError) as error:
        link_disjoint_union((diagram, diagram))

    assert (
        error.value.errors()[0]["type"] == "link_diagram.disjoint_union_crossing_bound"
    )


def test_disjoint_union_rejects_an_aggregate_free_loop_overflow() -> None:
    unknot = braid_closure(BraidWord(strand_count=2)).diagram
    looping = unknot.model_copy(update={"free_loops": MAX_LINK_CROSSINGS // 2 + 1})

    with pytest.raises(OperationResourceAdmissionError) as error:
        link_disjoint_union((looping, looping))

    assert (
        error.value.errors()[0]["type"] == "link_diagram.disjoint_union_free_loop_bound"
    )


def test_disjoint_union_refuses_a_forged_transport_map() -> None:
    """The carrier must not accept a transport that drops a source crossing."""
    result = _union()
    payload = result.model_dump()
    payload["crossing_map"] = payload["crossing_map"][:-1]

    with pytest.raises(ValueError):
        LinkDisjointUnionResult.model_validate(payload)


def test_disjoint_union_sign_preservation_is_observable() -> None:
    """Crossing metadata must survive the union, including inverse signs."""
    hopf = braid_closure(_hopf()).diagram
    result = _union()

    crossings = {
        crossing.crossing_id: crossing for crossing in result.diagram.crossings
    }
    for row in result.crossing_map:
        source = next(
            crossing
            for crossing in result.sources[row.source_index].crossings
            if crossing.crossing_id == row.source_crossing_id
        )
        target = crossings[row.target_crossing_id]
        assert (target.over_pair, target.under_pair, target.sign) == (
            source.over_pair,
            source.under_pair,
            source.sign,
        )
    assert {crossing.sign for crossing in hopf.crossings} == {1, -1}
