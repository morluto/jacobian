"""Exact bounded shortest-path analysis of complete Petri reachability graphs."""

from collections import deque

from jacobian.canonical import strict_json_object_size
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import (
    ReachabilityResult,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    _preflight_terminal_scc_graph_shape,
    _terminal_scc_adjacency,
    _validate_terminal_scc_net_values,
)
from jacobian.math.logic.automata.petri_nets.shortest_paths._models import (
    MAX_SHORTEST_PATH_LENGTH,
    MAX_SHORTEST_PATH_SEQUENCES,
    ShortestFiringSequencesResult,
)
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    FiringSequence,
    Marking,
)

MAX_SHORTEST_PATH_WORK = 20_000_000
MAX_SHORTEST_PATH_OUTPUT_BYTES = 10 * 1024 * 1024


def _admit_source_graph(
    graph: ReachabilityResult,
) -> tuple[int, dict[tuple[int, ...], int]]:
    places, transitions, label_characters, parented_markings = (
        _preflight_terminal_scc_graph_shape(graph)
    )
    state_count = len(graph.states)
    edge_count = len(graph.edges)
    base_labels = (
        *(graph.net.place_ids or ()),
        *(graph.net.transition_ids or ()),
    )
    work = (
        state_count * max(1, transitions) * max(1, places)
        + edge_count * max(1, places)
        + state_count
        + edge_count
        + parented_markings * places * transitions
        + label_characters
        + parented_markings * sum(len(label) for label in base_labels)
        + 4 * state_count * max(1, (state_count - 1).bit_length())
        + edge_count * max(1, (transitions - 1).bit_length())
    )
    if work > MAX_SHORTEST_PATH_WORK:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.shortest_sequences.work_bound",
            message="shortest-sequence graph validation exceeds its admitted work bound",
        )
    if graph.truncated:
        raise OperationDomainValidationError(
            location=("source_graph", "truncated"),
            code="petri_net.shortest_sequences.incomplete_graph",
            message="all shortest sequences require a complete reachability graph",
        )
    if graph.initial_marking.net is not None and graph.initial_marking.net != graph.net:
        raise OperationDomainValidationError(
            location=("source_graph", "initial_marking"),
            code="petri_net.shortest_sequences.marking_parent",
            message="source initial marking belongs to a different Petri net",
        )
    if any(
        state.marking.net is not None and state.marking.net != graph.net
        for state in graph.states
    ):
        raise OperationDomainValidationError(
            location=("source_graph", "states"),
            code="petri_net.shortest_sequences.marking_parent",
            message="source graph state belongs to a different Petri net",
        )
    _validate_terminal_scc_net_values(graph)
    indices: dict[tuple[int, ...], int] = {}
    for state in graph.states:
        tokens = state.marking.tokens
        if tokens in indices:
            raise OperationDomainValidationError(
                location=("source_graph", "states"),
                code="petri_net.shortest_sequences.duplicate_state",
                message="source graph must contain each marking exactly once",
            )
        indices[tokens] = state.state_index
    if graph.states[0].marking.tokens != graph.initial_marking.tokens:
        raise OperationDomainValidationError(
            location=("source_graph", "states"),
            code="petri_net.shortest_sequences.initial_state",
            message="state zero must be the source graph initial marking",
        )
    _terminal_scc_adjacency(graph, graph.net, indices)
    return work, indices


def _source_context_size(
    source_graph: ReachabilityResult, target_marking: Marking
) -> int:
    net_size = len(source_graph.net.model_dump_json().encode("utf-8"))
    initial_size = len(source_graph.initial_marking.model_dump_json().encode("utf-8"))
    target_size = len(target_marking.model_dump_json().encode("utf-8"))
    return strict_json_object_size(
        (
            ("net", net_size),
            ("initial_marking", initial_size),
            ("target_marking", target_size),
            ("shortest_length", 4),  # null or an admitted decimal integer
            ("sequences", 2),
        )
    )


def _labeled_adjacency(graph: ReachabilityResult) -> list[list[tuple[int, int]]]:
    labeled: list[list[tuple[int, int]]] = [[] for _ in graph.states]
    for edge in graph.edges:
        labeled[edge.source_state].append((edge.transition, edge.target_state))
    for outgoing in labeled:
        outgoing.sort()
    return labeled


def _shortest_distances(labeled: list[list[tuple[int, int]]]) -> list[int]:
    distance = [-1] * len(labeled)
    distance[0] = 0
    queue = deque([0])
    while queue:
        source = queue.popleft()
        for _, successor in labeled[source]:
            if distance[successor] == -1:
                distance[successor] = distance[source] + 1
                queue.append(successor)
    return distance


def _count_shortest_paths(
    labeled: list[list[tuple[int, int]]],
    distance: list[int],
    target: int,
    maximum_sequences: int,
) -> tuple[list[int], list[int]]:
    counts = [0] * len(labeled)
    total_steps = [0] * len(labeled)
    shortest_length = distance[target]
    count_cap = maximum_sequences + 1
    counts[target] = 1
    for state in sorted(range(len(labeled)), key=distance.__getitem__, reverse=True):
        if state == target or distance[state] < 0 or distance[state] >= shortest_length:
            continue
        for _, successor in labeled[state]:
            if (
                distance[successor] == distance[state] + 1
                and distance[successor] <= shortest_length
            ):
                counts[state] = min(count_cap, counts[state] + counts[successor])
                total_steps[state] = min(
                    MAX_SHORTEST_PATH_OUTPUT_BYTES + 1,
                    total_steps[state] + counts[successor] + total_steps[successor],
                )
    return counts, total_steps


def _enumerate_shortest_paths(
    labeled: list[list[tuple[int, int]]],
    distance: list[int],
    counts: list[int],
    target: int,
) -> tuple[FiringSequence, ...]:
    sequences: list[FiringSequence] = []
    path: list[int] = []
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        state, next_edge = stack[-1]
        if state == target:
            sequences.append(FiringSequence.model_construct(transitions=tuple(path)))
            stack.pop()
            if path:
                path.pop()
            continue
        outgoing = labeled[state]
        while next_edge < len(outgoing):
            transition, successor = outgoing[next_edge]
            next_edge += 1
            stack[-1] = (state, next_edge)
            if distance[successor] == distance[state] + 1 and counts[successor]:
                path.append(transition)
                stack.append((successor, 0))
                break
        else:
            stack.pop()
            if path:
                path.pop()
    return tuple(sequences)


def _shortest_prefix_counts(
    labeled: list[list[tuple[int, int]]],
    distance: list[int],
    counts: list[int],
    target: int,
) -> list[int]:
    """Count prefixes that can still reach the target in the shortest DAG."""
    prefixes = [0] * len(labeled)
    prefixes[0] = 1
    for state in sorted(range(len(labeled)), key=distance.__getitem__):
        if not prefixes[state] or state == target:
            continue
        for _, successor in labeled[state]:
            if counts[successor] and distance[successor] == distance[state] + 1:
                prefixes[successor] += prefixes[state]
    return prefixes


def shortest_firing_sequences(
    source_graph: ReachabilityResult, target_marking: Marking
) -> ShortestFiringSequencesResult:
    """Return every shortest transition word in a complete finite graph.

    Reachability graphs carry transition labels, so edges with equal endpoints
    but different transition indices remain distinct paths and distinct words.
    """
    if not isinstance(target_marking, Marking):
        raise OperationDomainValidationError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.target_type",
            message="target_marking must be a Marking value",
        )
    work, state_indices = _admit_source_graph(source_graph)
    if (target_marking.net is not None and target_marking.net != source_graph.net) or (
        not isinstance(target_marking.tokens, tuple)
        or len(target_marking.tokens) != source_graph.net.place_count
        or any(
            type(token) is not int or not 0 <= token <= MAX_PETRI_MARKING
            for token in target_marking.tokens
        )
    ):
        raise OperationDomainValidationError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.target_axis",
            message="target marking must belong to the source graph place axis",
        )

    try:
        source_graph = ReachabilityResult.model_validate(
            source_graph.model_dump(), strict=True
        )
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source_graph",),
            code="petri_net.shortest_sequences.graph_shape",
            message="source graph must satisfy its bounded canonical axes",
        ) from exc

    context_size = _source_context_size(source_graph, target_marking)
    if context_size > MAX_SHORTEST_PATH_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.shortest_sequences.output_bound",
            message="shortest-sequence context exceeds its serialized output bound",
        )

    target = state_indices.get(target_marking.tokens)
    if target is None:
        return ShortestFiringSequencesResult._from_kernel(
            net=source_graph.net,
            initial_marking=source_graph.initial_marking,
            target_marking=target_marking,
            shortest_length=None,
            sequences=(),
        )

    labeled = _labeled_adjacency(source_graph)
    distance = _shortest_distances(labeled)
    shortest_length = distance[target]
    if shortest_length < 0:
        raise OperationDomainValidationError(
            location=("source_graph", "states"),
            code="petri_net.shortest_sequences.unreachable_state",
            message="validated source graph must make every represented marking reachable",
        )
    if shortest_length > MAX_SHORTEST_PATH_LENGTH:
        raise OperationResourceAdmissionError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.length_bound",
            message="shortest firing sequences exceed the replayable sequence bound",
        )

    # Include the exact net and both marking contexts, then bound each path by
    # two-digit transition indices, separators, and the typed-object wrapper.
    output_per_sequence = 24 + 3 * shortest_length
    maximum_sequences = min(
        MAX_SHORTEST_PATH_SEQUENCES,
        max(
            0,
            (MAX_SHORTEST_PATH_OUTPUT_BYTES - context_size - 64) // output_per_sequence,
        ),
    )
    counts, total_steps = _count_shortest_paths(
        labeled, distance, target, maximum_sequences
    )
    if counts[0] == 0:
        raise OperationDomainValidationError(
            location=("source_graph", "edges"),
            code="petri_net.shortest_sequences.path_dag",
            message="validated shortest-distance graph must contain a path to the target",
        )
    if counts[0] > maximum_sequences:
        raise OperationResourceAdmissionError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.output_bound",
            message="complete shortest-sequence family exceeds its admitted output bound",
        )
    estimate = context_size + 64 + counts[0] * output_per_sequence
    if estimate > MAX_SHORTEST_PATH_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.output_bound",
            message="complete shortest-sequence family exceeds its admitted output bound",
        )
    # Enumeration scans each state's whole outgoing list once per shortest
    # prefix reaching it; count those scans before materializing the family.
    prefix_counts = _shortest_prefix_counts(labeled, distance, counts, target)
    scan_work = sum(
        prefix_counts[state] * len(labeled[state])
        for state in range(len(labeled))
        if state != target and prefix_counts[state]
    )
    if work + total_steps[0] + scan_work > MAX_SHORTEST_PATH_WORK:
        raise OperationResourceAdmissionError(
            location=("target_marking",),
            code="petri_net.shortest_sequences.work_bound",
            message="shortest-sequence enumeration exceeds its admitted work bound",
        )

    sequence_values = _enumerate_shortest_paths(labeled, distance, counts, target)
    return ShortestFiringSequencesResult._from_kernel(
        net=source_graph.net,
        initial_marking=source_graph.initial_marking,
        target_marking=target_marking,
        shortest_length=shortest_length,
        sequences=sequence_values,
    )
