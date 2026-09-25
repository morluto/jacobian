"""Contracts for terminal SCCs of bounded Petri reachability graphs."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets import operations as petri_operations
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.operations import (
    reachability_graph,
    reachability_terminal_scc_profile,
)
from jacobian.math.logic.automata.petri_nets.values import (
    Marking,
    PetriMarkingState,
    PetriNet,
    PetriReachabilityEdge,
)


def test_finds_distinct_terminal_components_and_keeps_source_graph() -> None:
    net = PetriNet(
        place_count=3,
        transition_count=4,
        pre=((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        post=((0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)),
    )
    initial = Marking(tokens=(1, 0, 0))
    graph = reachability_graph(net, initial, max_states=8)

    result = reachability_terminal_scc_profile(graph)

    assert result.terminal_components == ((1,), (2,))
    assert result.source_graph == graph
    assert result.model_validate_json(result.model_dump_json()) == result


def test_cycle_is_one_terminal_scc_and_dead_state_is_singleton() -> None:
    cycle_net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    cycle_graph = reachability_graph(cycle_net, Marking(tokens=(1, 0)), max_states=8)
    assert reachability_terminal_scc_profile(cycle_graph).terminal_components == (
        (0, 1),
    )

    dead_net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    dead_graph = reachability_graph(dead_net, Marking(tokens=(1,)), max_states=8)
    assert reachability_terminal_scc_profile(dead_graph).terminal_components == ((1,),)


def test_truncated_graph_only_claims_sinks_of_the_represented_subgraph() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=1,
        pre=((1,), (0,)),
        post=((0,), (1,)),
    )
    graph = reachability_graph(net, Marking(tokens=(1, 0)), max_states=1)
    assert graph.truncated is True
    result = reachability_terminal_scc_profile(graph)
    assert result.terminal_components == ((0,),)
    assert result.source_graph.truncated is True


def test_rejects_illegal_edge_and_incomplete_untruncated_graph() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    graph = reachability_graph(net, Marking(tokens=(1, 0)), max_states=4)
    incomplete = graph.model_copy(update={"edges": graph.edges[:-1]})
    with pytest.raises(OperationDomainValidationError, match="every enabled successor"):
        reachability_terminal_scc_profile(incomplete)

    forged = graph.model_copy(
        update={
            "edges": (
                PetriReachabilityEdge(source_state=0, transition=0, target_state=0),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError, match="exact firing"):
        reachability_terminal_scc_profile(forged)


def test_rejects_empty_source_graph_before_scc_construction() -> None:
    net = PetriNet(
        place_count=0,
        transition_count=0,
        pre=(),
        post=(),
    )
    empty_graph = ReachabilityResult.model_construct(
        net=net,
        initial_marking=Marking(tokens=()),
        max_states=1,
        states=(),
        edges=(),
        truncated=False,
    )
    with pytest.raises(OperationDomainValidationError, match="state/edge counts"):
        reachability_terminal_scc_profile(empty_graph)


def test_rejects_forged_oversized_matrix_before_model_dump() -> None:
    oversized_net = PetriNet.model_construct(
        place_count=1,
        transition_count=1,
        pre=((0,) * 100_000,),
        post=((0,),),
    )
    graph = ReachabilityResult.model_construct(
        net=oversized_net,
        initial_marking=Marking.model_construct(tokens=(0,)),
        max_states=1,
        states=(),
        edges=(),
        truncated=True,
    )
    with pytest.raises(OperationDomainValidationError, match="arc matrices"):
        reachability_terminal_scc_profile(graph)


def test_work_bound_rejects_before_graph_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    size = 64
    net = PetriNet(
        place_count=size,
        transition_count=size,
        pre=tuple((0,) * size for _ in range(size)),
        post=tuple((0,) * size for _ in range(size)),
    )
    marking = Marking.model_construct(tokens=(0,) * size)
    states = tuple(
        PetriMarkingState.model_construct(
            state_index=index,
            place_axis=tuple(range(size)),
            marking=marking,
        )
        for index in range(250)
    )
    graph = ReachabilityResult.model_construct(
        net=net,
        initial_marking=marking,
        max_states=250,
        states=states,
        edges=(),
        truncated=True,
    )

    def fail_on_dump(self: ReachabilityResult, *args: object, **kwargs: object) -> None:
        pytest.fail("graph serialization ran before work admission")

    monkeypatch.setattr(ReachabilityResult, "model_dump", fail_on_dump)
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        reachability_terminal_scc_profile(graph)


def test_ordering_work_is_included_at_the_exact_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    graph = reachability_graph(net, Marking(tokens=(1,)), max_states=2)
    state_count = len(graph.states)
    assert state_count == 2
    parented_markings = sum(
        marking.net is not None
        for marking in (
            graph.initial_marking,
            *(state.marking for state in graph.states),
        )
    )
    base_work = (
        state_count
        + len(graph.edges)
        + state_count
        + len(graph.edges)
        + 1
        + parented_markings
    )
    ordering_work = 4 * state_count * (state_count - 1).bit_length()
    monkeypatch.setattr(
        petri_operations, "MAX_TERMINAL_SCC_PROFILE_WORK", base_work + ordering_work
    )
    assert reachability_terminal_scc_profile(graph).terminal_components == ((1,),)

    monkeypatch.setattr(
        petri_operations, "MAX_TERMINAL_SCC_PROFILE_WORK", base_work + ordering_work - 1
    )
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        reachability_terminal_scc_profile(graph)


def test_charges_each_parent_markings_own_matrix_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    parent_net = PetriNet(
        place_count=64,
        transition_count=64,
        pre=tuple((0,) * 64 for _ in range(64)),
        post=tuple((0,) * 64 for _ in range(64)),
    )
    marking = Marking.model_construct(tokens=(0,), net=parent_net)
    state_count = 250
    states = tuple(
        PetriMarkingState.model_construct(
            state_index=index,
            place_axis=(0,),
            marking=marking,
        )
        for index in range(state_count)
    )
    graph = ReachabilityResult.model_construct(
        net=source_net,
        initial_marking=marking,
        max_states=state_count,
        states=states,
        edges=(),
        truncated=True,
    )

    def fail_on_output_bound(*args: object, **kwargs: object) -> None:
        pytest.fail("output-bound traversal ran before work admission")

    monkeypatch.setattr(
        petri_operations, "_terminal_scc_graph_output_bound", fail_on_output_bound
    )
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        reachability_terminal_scc_profile(graph)


def test_preserves_domain_classification_for_malformed_graphs() -> None:
    source_net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    for bad_net, expected_code in (
        ("not-a-petri-net", "petri_net.terminal_scc.net_type"),
        (
            PetriNet.model_construct(
                place_count="1", transition_count=1, pre=((0,),), post=((0,),)
            ),
            "petri_net.terminal_scc.net_axes",
        ),
    ):
        graph = ReachabilityResult.model_construct(
            net=bad_net,
            initial_marking=Marking.model_construct(tokens=(0,)),
            max_states=1,
            states=(),
            edges=(),
            truncated=True,
        )
        with pytest.raises(OperationDomainValidationError) as excinfo:
            reachability_terminal_scc_profile(graph)
        assert excinfo.value.errors()[0]["type"] == expected_code

    list_shaped = ReachabilityResult.model_construct(
        net=source_net,
        initial_marking=Marking.model_construct(tokens=(0,)),
        max_states=1,
        states=[],
        edges=[],
        truncated=True,
    )
    with pytest.raises(OperationDomainValidationError) as excinfo:
        reachability_terminal_scc_profile(list_shaped)
    assert excinfo.value.errors()[0]["type"] == "petri_net.terminal_scc.graph_shape"
