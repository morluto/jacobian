"""Exact disjoint-union laws for bounded weighted Petri nets."""

from __future__ import annotations

from collections import deque

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    disjoint_union,
    fire_transition,
    reachability_graph,
)
from jacobian.math.logic.automata.petri_nets._models import (
    PetriNetDisjointUnionRequest,
)
from jacobian.math.logic.automata.petri_nets._tools import TOOLS


def _cycle() -> PetriNet:
    return PetriNet(
        place_count=2,
        transition_count=2,
        place_ids=("a", "b"),
        transition_ids=("ab", "ba"),
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )


def test_marked_disjoint_union_is_block_diagonal_and_embeds_each_firing() -> None:
    left = _cycle()
    right = _cycle()
    left_marking = Marking(tokens=(1, 0), net=left)
    right_marking = Marking(tokens=(0, 1), net=right)

    result = disjoint_union(left, right, left_marking, right_marking)

    assert result.net.pre == (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )
    assert result.net.post == (
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
    )
    assert result.net.place_ids == ("L:0:a", "L:1:b", "R:0:a", "R:1:b")
    assert result.net.transition_ids == ("L:0:ab", "L:1:ba", "R:0:ab", "R:1:ba")
    assert result.left_place_embedding == (0, 1)
    assert result.right_place_embedding == (2, 3)
    assert result.left_transition_embedding == (0, 1)
    assert result.right_transition_embedding == (2, 3)
    assert result.marking == Marking(tokens=(1, 0, 0, 1), net=result.net)

    left_step = fire_transition(left, left_marking, 0)
    union_step = fire_transition(result.net, result.marking, 0)
    assert left_step.new_marking is not None
    assert union_step.new_marking is not None
    assert (
        union_step.new_marking.tokens
        == left_step.new_marking.tokens + right_marking.tokens
    )


def _reference_reachability(
    net: PetriNet, initial: tuple[int, ...]
) -> tuple[set[tuple[int, ...]], set[tuple[tuple[int, ...], int, tuple[int, ...]]]]:
    """Enumerate a small finite P/T graph directly from the firing rule."""
    states = {initial}
    edges: set[tuple[tuple[int, ...], int, tuple[int, ...]]] = set()
    queue = deque((initial,))
    while queue:
        marking = queue.popleft()
        for transition in range(net.transition_count):
            if any(
                marking[place] < net.pre[place][transition]
                for place in range(net.place_count)
            ):
                continue
            successor = tuple(
                marking[place]
                - net.pre[place][transition]
                + net.post[place][transition]
                for place in range(net.place_count)
            )
            edges.add((marking, transition, successor))
            if successor not in states:
                states.add(successor)
                queue.append(successor)
    return states, edges


def test_small_reachability_graph_of_union_is_cartesian_product() -> None:
    left = _cycle()
    right = _cycle()
    lm = Marking(tokens=(1, 0), net=left)
    rm = Marking(tokens=(0, 1), net=right)
    union = disjoint_union(left, right, lm, rm)
    assert union.marking is not None

    left_states, left_edges = _reference_reachability(left, lm.tokens)
    right_states, right_edges = _reference_reachability(right, rm.tokens)
    expected_states = {a + b for a in left_states for b in right_states}
    expected_edges = {
        (source + other, transition, target + other)
        for source, transition, target in left_edges
        for other in right_states
    } | {
        (other + source, left.transition_count + transition, other + target)
        for source, transition, target in right_edges
        for other in left_states
    }

    graph = reachability_graph(union.net, union.marking, max_states=4)
    state_by_index = {state.state_index: state.marking.tokens for state in graph.states}
    actual_states = set(state_by_index.values())
    actual_edges = {
        (
            state_by_index[edge.source_state],
            edge.transition,
            state_by_index[edge.target_state],
        )
        for edge in graph.edges
    }
    assert not graph.truncated
    assert actual_states == expected_states
    assert actual_edges == expected_edges


def test_empty_axes_and_optional_markings_keep_canonical_values() -> None:
    empty = PetriNet(place_count=0, transition_count=0, pre=(), post=())
    net = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    result = disjoint_union(empty, net)
    assert result.net == net
    assert result.marking is None
    assert disjoint_union(
        empty, net, Marking(tokens=()), Marking(tokens=(3,))
    ).marking == Marking(tokens=(3,), net=net)


def test_markings_must_be_supplied_as_a_pair_and_axes_are_bound() -> None:
    left = _cycle()
    foreign = PetriNet(place_count=2, transition_count=2, pre=left.pre, post=left.post)
    with pytest.raises(ValueError):
        PetriNetDisjointUnionRequest(
            left_net=left, right_net=left, left_marking=Marking(tokens=(1, 0))
        )
    with pytest.raises(ValueError):
        disjoint_union(
            left, left, Marking(tokens=(1, 0), net=foreign), Marking(tokens=(1, 0))
        )


def test_union_accepts_the_full_carrier_matrix_boundary() -> None:
    size = 32
    zeros = (0,) * size
    net = PetriNet(
        place_count=size,
        transition_count=size,
        pre=(zeros,) * size,
        post=(zeros,) * size,
    )
    result = disjoint_union(net, net)
    assert result.net.place_count == 64
    assert result.net.transition_count == 64
    assert len(result.net.pre) == 64
    assert all(len(row) == 64 and not any(row) for row in result.net.pre)


def test_union_rejects_combined_axes_outside_the_carrier_before_matrix_build() -> None:
    wide = PetriNet(place_count=64, transition_count=0, pre=((),) * 64, post=((),) * 64)
    one_place = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    with pytest.raises(OperationResourceAdmissionError, match="disjoint union exceeds"):
        disjoint_union(wide, one_place)


def test_disjoint_union_manifest_declares_typed_pair_operation() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "petri_net.disjoint_union.compute"
    )
    assert tool.request_type is PetriNetDisjointUnionRequest
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.marking is not None
    assert result.marking.tokens == (1, 2)
    assert result.net.place_ids == ("L:0:buffer", "R:0:buffer")
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_marked_union_accounts_for_recursively_serialized_parent_nets() -> None:
    # JSON escaping expands each control code to six bytes; all three marking
    # parents (two source nets and the union net) are serialized recursively.
    oversized_label = "\\x01" * 700_000
    net = PetriNet(
        place_count=1,
        transition_count=0,
        place_ids=(oversized_label,),
        pre=((),),
        post=((),),
    )
    marking = Marking(tokens=(0,), net=net)
    with pytest.raises(
        OperationResourceAdmissionError, match="serialized-output bound"
    ):
        disjoint_union(net, net, marking, marking)


def test_output_growth_is_rejected_before_union_matrix_materialization() -> None:
    oversized_label = "x" * 1_800_000
    net = PetriNet(
        place_count=1,
        transition_count=0,
        place_ids=(oversized_label,),
        pre=((),),
        post=((),),
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="serialized-output bound"
    ):
        disjoint_union(
            net, PetriNet(place_count=0, transition_count=0, pre=(), post=())
        )


def test_malformed_axis_encoding_precedes_union_resource_admission() -> None:
    # An unpaired surrogate is a canonical-encoding domain failure.  The same
    # malformed axis must not change error category when its serialized size
    # crosses the resource envelope.
    empty = PetriNet(place_count=0, transition_count=0, pre=(), post=())

    short = PetriNet(
        place_count=1,
        transition_count=0,
        place_ids=("\ud800",),
        pre=((),),
        post=((),),
    )
    with pytest.raises(OperationDomainValidationError) as short_error:
        disjoint_union(short, empty)
    assert short_error.value.errors()[0]["type"] == "petri_net.net_axis_encoding"

    oversized = PetriNet(
        place_count=1,
        transition_count=0,
        place_ids=("\ud800" * 1_800_000,),
        pre=((),),
        post=((),),
    )
    with pytest.raises(OperationDomainValidationError) as long_error:
        disjoint_union(oversized, empty)
    assert not isinstance(long_error.value, OperationResourceAdmissionError)
    assert long_error.value.errors()[0]["type"] == "petri_net.net_axis_encoding"
