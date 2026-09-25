"""All shortest Petri firing words, checked against direct firing semantics."""

from collections import deque

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.automata.petri_nets.operations import reachability_graph
from jacobian.math.logic.automata.petri_nets.shortest_paths.operations import (
    shortest_firing_sequences,
)
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


def _oracle(
    net: PetriNet, initial: tuple[int, ...], target: tuple[int, ...]
) -> tuple[int | None, tuple[tuple[int, ...], ...]]:
    """Enumerate shortest words from the enabling and firing definitions."""
    queue = deque([(initial, ())])
    best_distance: dict[tuple[int, ...], int] = {initial: 0}
    answers: list[tuple[int, ...]] = []
    target_distance: int | None = None
    while queue:
        marking, word = queue.popleft()
        if len(word) > 20 or len(best_distance) > 100 or len(queue) > 10_000:
            raise AssertionError("test oracle exceeded its small finite-state envelope")
        if target_distance is not None and len(word) > target_distance:
            break
        if marking == target:
            target_distance = len(word)
            answers.append(word)
            continue
        for transition in range(net.transition_count):
            if any(marking[p] < net.pre[p][transition] for p in range(net.place_count)):
                continue
            successor = tuple(
                marking[p] - net.pre[p][transition] + net.post[p][transition]
                for p in range(net.place_count)
            )
            next_word = (*word, transition)
            distance = len(next_word)
            previous = best_distance.get(successor)
            if previous is None or distance < previous:
                best_distance[successor] = distance
                queue.append((successor, next_word))
            elif distance == previous:
                # Retain equal-distance words: they are distinct paths.
                queue.append((successor, next_word))
    if target_distance is None:
        return None, ()
    return target_distance, tuple(sorted(set(answers)))


def test_all_shortest_sequences_match_direct_firing_oracle() -> None:
    # Two labelled transitions reach the same marking in one step. A longer
    # self-loop is added at the target and must not appear in the shortest set.
    net = PetriNet(
        place_count=2,
        transition_count=3,
        pre=((1, 1, 0), (0, 0, 1)),
        post=((0, 0, 0), (1, 1, 1)),
    )
    initial = Marking(tokens=(1, 0))
    target = Marking(tokens=(0, 1))
    graph = reachability_graph(net, initial, max_states=3)
    expected_length, expected = _oracle(net, initial.tokens, target.tokens)

    result = shortest_firing_sequences(graph, target)

    assert result.shortest_length == expected_length == 1
    assert (
        tuple(item.transitions for item in result.sequences) == expected == ((0,), (1,))
    )


def test_initial_and_unreachable_targets_have_exact_empty_family_conventions() -> None:
    net = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    initial = Marking(tokens=(0,))
    graph = reachability_graph(net, initial, max_states=1)

    at_initial = shortest_firing_sequences(graph, initial)
    unreachable = shortest_firing_sequences(graph, Marking(tokens=(1,)))

    assert at_initial.shortest_length == 0
    assert tuple(item.transitions for item in at_initial.sequences) == ((),)
    assert unreachable.shortest_length is None
    assert unreachable.sequences == ()


def test_all_shortest_words_on_a_two_branch_diamond() -> None:
    net = PetriNet(
        place_count=4,
        transition_count=4,
        pre=((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0)),
        post=((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 1)),
    )
    initial = Marking(tokens=(1, 0, 0, 0))
    target = Marking(tokens=(0, 0, 0, 1))
    graph = reachability_graph(net, initial, max_states=4)
    expected_length, expected = _oracle(net, initial.tokens, target.tokens)

    result = shortest_firing_sequences(graph, target)

    assert result.shortest_length == expected_length == 2
    assert (
        tuple(item.transitions for item in result.sequences)
        == expected
        == (
            (0, 2),
            (1, 3),
        )
    )


def test_truncated_graph_cannot_claim_complete_shortest_family() -> None:
    net = PetriNet(place_count=1, transition_count=1, pre=((0,),), post=((1,),))
    graph = reachability_graph(net, Marking(tokens=(0,)), max_states=1)
    assert graph.truncated

    with pytest.raises(
        OperationDomainValidationError, match="complete reachability graph"
    ):
        shortest_firing_sequences(graph, Marking(tokens=(0,)))


def test_catalog_example_preserves_parallel_transition_labels() -> None:
    catalog = Catalog.open()
    operation_id = "petri_net.reachability.shortest_sequences.compute"
    result = invoke_operation(
        operation_id,
        catalog.operation(operation_id).examples[0].input,
        catalog,
    )

    assert result.output["shortest_length"] == 1
    assert [path["transitions"] for path in result.output["sequences"]] == [[0], [1]]
