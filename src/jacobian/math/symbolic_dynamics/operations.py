"""Exact bounded native kernels for symbolic dynamics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from jacobian.math.symbolic_dynamics.values import (
    MAX_ENUMERATED_BLOCKS,
    MAX_PRESENTATION_CELLS,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
    LabeledTransition,
)


def _contains(word: tuple[str, ...], factor: tuple[str, ...]) -> bool:
    return any(
        word[start : start + len(factor)] == factor
        for start in range(len(word) - len(factor) + 1)
    )


def enumeration_size(alphabet_size: int, block_length: int) -> int:
    if block_length < 0:
        raise ValueError("block length must be nonnegative")
    size = 1
    for _ in range(block_length):
        size *= alphabet_size
        if size > MAX_ENUMERATED_BLOCKS:
            raise ValueError("requested block enumeration exceeds the work bound")
    return size


def normalize_forbidden_blocks(
    shift: ForbiddenBlockShift,
) -> tuple[tuple[str, ...], ...]:
    rank = {symbol: index for index, symbol in enumerate(shift.alphabet)}
    ordered = sorted(
        set(shift.forbidden_blocks),
        key=lambda block: (len(block), tuple(rank[symbol] for symbol in block)),
    )
    minimal: list[tuple[str, ...]] = []
    for block in ordered:
        if not any(_contains(block, forbidden) for forbidden in minimal):
            minimal.append(block)
    return tuple(minimal)


def _presentation_memory(shift: ForbiddenBlockShift) -> int:
    return max(
        0,
        max(
            (len(block) - 1 for block in normalize_forbidden_blocks(shift)),
            default=0,
        ),
    )


def _require_bounded_support(shift: ForbiddenBlockShift) -> int:
    memory = _presentation_memory(shift)
    enumeration_size(len(shift.alphabet), memory + 1)
    return memory


def _require_bounded_presentation(shift: ForbiddenBlockShift, memory: int) -> None:
    state_count = enumeration_size(len(shift.alphabet), memory)
    enumeration_size(len(shift.alphabet), memory + 1)
    if state_count * state_count > MAX_PRESENTATION_CELLS:
        raise ValueError("presentation adjacency exceeds the result bound")


def block_language(
    shift: ForbiddenBlockShift, block_length: int
) -> tuple[tuple[str, ...], ...]:
    if block_length < 0:
        raise ValueError("block length must be nonnegative")
    enumeration_size(len(shift.alphabet), block_length)
    forbidden = normalize_forbidden_blocks(shift)
    if () in forbidden:
        return ()
    memory = _require_bounded_support(shift)
    states, _, left_infinite, right_infinite = _presentation_support(shift, memory)
    if block_length == 0:
        return (
            ((),)
            if right_infinite
            and (not shift.two_sided or left_infinite & right_infinite)
            else ()
        )
    candidates = _locally_allowed_words(shift, block_length)
    if memory == 0:
        return candidates if right_infinite else ()
    if block_length < memory:
        supported_states = (
            left_infinite & right_infinite if shift.two_sided else right_infinite
        )
        supported_prefixes = {state[:block_length] for state in supported_states}
        return tuple(word for word in candidates if word in supported_prefixes)
    state_index = set(states)
    occurring = []
    for word in candidates:
        starts = (word[:memory],)
        ends = (word[-memory:],)
        if any(
            start in state_index
            and end in right_infinite
            and (not shift.two_sided or start in left_infinite)
            for start in starts
            for end in ends
        ):
            occurring.append(word)
    return tuple(occurring)


def _locally_allowed_words(
    shift: ForbiddenBlockShift, length: int
) -> tuple[tuple[str, ...], ...]:
    forbidden = normalize_forbidden_blocks(shift)
    return tuple(
        word
        for word in itertools.product(shift.alphabet, repeat=length)
        if not any(_contains(word, excluded) for excluded in forbidden)
    )


def _presentation_support(
    shift: ForbiddenBlockShift, memory: int
) -> tuple[
    tuple[tuple[str, ...], ...],
    nx.DiGraph[tuple[str, ...]],
    set[tuple[str, ...]],
    set[tuple[str, ...]],
]:
    import networkx as nx

    states = _locally_allowed_words(shift, memory)
    extensions = _locally_allowed_words(shift, memory + 1)
    graph: nx.DiGraph[tuple[str, ...]] = nx.DiGraph()
    graph.add_nodes_from(states)
    for word in extensions:
        source = word[:memory] if memory else ()
        target = word[-memory:] if memory else ()
        if source in graph and target in graph:
            graph.add_edge(source, target)
    cyclic = {
        node
        for component in nx.strongly_connected_components(graph)
        for node in component
        if len(component) > 1 or graph.has_edge(node, node)
    }
    right_infinite = _reachable_from(graph, cyclic, reverse=True)
    left_infinite = _reachable_from(graph, cyclic, reverse=False)
    return states, graph, left_infinite, right_infinite


def _reachable_from(
    graph: nx.DiGraph[tuple[str, ...]],
    starts: set[tuple[str, ...]],
    *,
    reverse: bool,
) -> set[tuple[str, ...]]:
    reached = set(starts)
    pending = list(starts)
    while pending:
        node = pending.pop()
        neighbors = graph.predecessors(node) if reverse else graph.successors(node)
        for neighbor in neighbors:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _presentation_from_states_and_words(
    shift: ForbiddenBlockShift,
    memory: int,
    states: tuple[tuple[str, ...], ...],
    extension_words: tuple[tuple[str, ...], ...],
) -> BlockPresentation:
    state_index = {state: index for index, state in enumerate(states)}
    transitions: list[LabeledTransition] = []
    for word in extension_words:
        source_block = word[:memory] if memory else ()
        target_block = word[-memory:] if memory else ()
        source = state_index.get(source_block)
        target = state_index.get(target_block)
        if source is not None and target is not None:
            transitions.append(
                LabeledTransition(
                    source=source,
                    target=target,
                    appended_symbol=word[-1],
                )
            )
    size = len(states)
    adjacency = [[0] * size for _ in range(size)]
    for transition in transitions:
        adjacency[transition.source][transition.target] += 1
    return BlockPresentation(
        alphabet=shift.alphabet,
        memory=memory,
        state_blocks=states,
        transitions=tuple(transitions),
        adjacency_matrix=tuple(tuple(row) for row in adjacency),
        two_sided=shift.two_sided,
    )


def finite_type_presentation(shift: ForbiddenBlockShift) -> BlockPresentation:
    memory = _presentation_memory(shift)
    _require_bounded_presentation(shift, memory)
    states = block_language(shift, memory)
    extensions = block_language(shift, memory + 1)
    return _presentation_from_states_and_words(shift, memory, states, extensions)


def higher_block_presentation(
    shift: ForbiddenBlockShift, block_length: int
) -> BlockPresentation:
    if block_length < 1:
        raise ValueError("higher-block length must be positive")
    required_memory = _presentation_memory(shift)
    if block_length < required_memory:
        raise ValueError("higher-block length is below the presentation memory")
    _require_bounded_presentation(shift, block_length)
    states = block_language(shift, block_length)
    extensions = block_language(shift, block_length + 1)
    return _presentation_from_states_and_words(shift, block_length, states, extensions)


def adjacency_shift(
    matrix: tuple[tuple[int, ...], ...], *, two_sided: bool = True
) -> AdjacencyShift:
    return AdjacencyShift(matrix=matrix, two_sided=two_sided)


def _matrix_product(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _mobius(value: int) -> int:
    remaining = value
    prime = 2
    distinct_factors = 0
    while prime * prime <= remaining:
        if remaining % prime:
            prime += 1
            continue
        remaining //= prime
        distinct_factors += 1
        if remaining % prime == 0:
            return 0
        while remaining % prime == 0:
            remaining //= prime
        prime += 1
    if remaining > 1:
        distinct_factors += 1
    return -1 if distinct_factors % 2 else 1


def periodic_point_profile(
    shift: AdjacencyShift, max_period: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not 1 <= max_period <= 50:
        raise ValueError("max period is outside the supported bounds")
    matrix = shift.matrix
    power = matrix
    fixed: list[int] = []
    for period in range(1, max_period + 1):
        fixed.append(sum(power[index][index] for index in range(len(matrix))))
        if period < max_period:
            power = _matrix_product(power, matrix)
    exact = tuple(
        sum(
            _mobius(divisor) * fixed[period // divisor - 1]
            for divisor in range(1, period + 1)
            if period % divisor == 0
        )
        for period in range(1, max_period + 1)
    )
    if any(count < 0 or count % period for period, count in enumerate(exact, 1)):
        raise RuntimeError("periodic-point inversion violated orbit integrality")
    orbits = tuple(count // period for period, count in enumerate(exact, 1))
    return tuple(fixed), exact, orbits


__all__ = [
    "adjacency_shift",
    "block_language",
    "enumeration_size",
    "finite_type_presentation",
    "higher_block_presentation",
    "normalize_forbidden_blocks",
    "periodic_point_profile",
]
