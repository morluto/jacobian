"""Kernels for Petri-net transition liveness profiles."""

from jacobian.canonical import strict_json_object_size
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import (
    ReachabilityResult,
)
from jacobian.math.logic.automata.petri_nets.liveness._models import (
    MAX_TRANSITION_LIVENESS_OUTPUT_BYTES,
    MAX_TRANSITION_LIVENESS_WORK,
    TransitionLivenessEntry,
    TransitionLivenessResult,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    MAX_TERMINAL_SCC_PROFILE_OUTPUT_BYTES,
    MAX_TERMINAL_SCC_PROFILE_WORK,
    _preflight_terminal_scc_graph_shape,
    _terminal_scc_adjacency,
    _validate_terminal_scc_net_values,
)


def _output_size(graph: ReachabilityResult, transition_count: int) -> int:
    """Bound serialized output without allocating per-state/per-edge lists."""

    def array_size(count: int, item_sizes: int) -> int:
        return 2 + max(0, count - 1) + item_sizes

    def string_size(value: str) -> int:
        size = 2
        for character in value:
            codepoint = ord(character)
            if character in ('"', "\\") or character in "\b\t\n\f\r":
                size += 2
            elif codepoint <= 0x1F:
                size += 6
            elif 0xD800 <= codepoint <= 0xDFFF:
                raise OperationDomainValidationError(
                    location=("source_graph", "net"),
                    code="petri_net.transition_liveness.axis_encoding",
                    message="Petri-net axis IDs must be valid Unicode strings",
                )
            else:
                size += len(character.encode("utf-8"))
        return size

    def net_size(net: object) -> int:
        place_ids = (
            4
            if net.place_ids is None
            else array_size(
                len(net.place_ids), sum(string_size(item) for item in net.place_ids)
            )
        )
        transition_ids = (
            4
            if net.transition_ids is None
            else array_size(
                len(net.transition_ids),
                sum(string_size(item) for item in net.transition_ids),
            )
        )
        pre_size = array_size(
            len(net.pre),
            sum(
                array_size(len(row), sum(len(str(item)) for item in row))
                for row in net.pre
            ),
        )
        post_size = array_size(
            len(net.post),
            sum(
                array_size(len(row), sum(len(str(item)) for item in row))
                for row in net.post
            ),
        )
        return strict_json_object_size(
            (
                ("place_count", len(str(net.place_count))),
                ("transition_count", len(str(net.transition_count))),
                ("place_ids", place_ids),
                ("transition_ids", transition_ids),
                ("pre", pre_size),
                ("post", post_size),
            )
        )

    def marking_size(marking: object) -> int:
        tokens = array_size(
            len(marking.tokens), sum(len(str(token)) for token in marking.tokens)
        )
        parent = 4 if marking.net is None else net_size(marking.net)
        return strict_json_object_size((("tokens", tokens), ("net", parent)))

    states_size = array_size(
        len(graph.states),
        sum(
            strict_json_object_size(
                (
                    ("state_index", len(str(state.state_index))),
                    (
                        "place_axis",
                        array_size(
                            len(state.place_axis),
                            sum(len(str(i)) for i in state.place_axis),
                        ),
                    ),
                    ("marking", marking_size(state.marking)),
                )
            )
            for state in graph.states
        ),
    )
    edges_size = array_size(
        len(graph.edges),
        sum(
            strict_json_object_size(
                (
                    ("source_state", len(str(edge.source_state))),
                    ("transition", len(str(edge.transition))),
                    ("target_state", len(str(edge.target_state))),
                )
            )
            for edge in graph.edges
        ),
    )
    source_size = strict_json_object_size(
        (
            ("net", net_size(graph.net)),
            ("initial_marking", marking_size(graph.initial_marking)),
            ("max_states", len(str(graph.max_states))),
            ("states", states_size),
            ("edges", edges_size),
            ("truncated", 4 if graph.truncated else 5),
        )
    )
    # Each entry has a bounded index, a short status, and an optional index.
    profile_size = array_size(transition_count, transition_count * 96)
    return strict_json_object_size(
        (("source_graph", source_size), ("transitions", profile_size))
    )


def _classify_transitions(
    graph: ReachabilityResult,
    reverse: list[list[int]],
    transition_count: int,
) -> tuple[TransitionLivenessEntry, ...]:
    edge_sources: list[set[int]] = [set() for _ in range(transition_count)]
    for edge in graph.edges:
        edge_sources[edge.transition].add(edge.source_state)
    entries: list[TransitionLivenessEntry] = []
    state_count = len(graph.states)
    for transition in range(transition_count):
        if graph.truncated:
            entries.append(
                TransitionLivenessEntry.model_construct(
                    transition=transition, status="UNKNOWN", witness_state=None
                )
            )
            continue
        can_fire = edge_sources[transition]
        pending = list(can_fire)
        while pending:
            for predecessor in reverse[pending.pop()]:
                if predecessor not in can_fire:
                    can_fire.add(predecessor)
                    pending.append(predecessor)
        if len(can_fire) == state_count:
            entries.append(
                TransitionLivenessEntry.model_construct(
                    transition=transition, status="LIVE", witness_state=None
                )
            )
        else:
            witness = next(
                state for state in range(state_count) if state not in can_fire
            )
            entries.append(
                TransitionLivenessEntry.model_construct(
                    transition=transition,
                    status="NOT_LIVE",
                    witness_state=witness,
                )
            )
    return tuple(entries)


def transition_liveness_profile(
    source_graph: ReachabilityResult,
) -> TransitionLivenessResult:
    """Classify each transition under standard Petri-net L4 liveness.

    A transition is LIVE exactly when, from every reachable marking, some
    continuation can fire it. This uses a complete finite graph. A truncated
    graph returns UNKNOWN for every transition, including observed self-loops.
    """
    if not isinstance(source_graph, ReachabilityResult):
        raise OperationDomainValidationError(
            location=("source_graph",),
            code="petri_net.transition_liveness.graph_type",
            message="source_graph must be a ReachabilityResult",
        )
    # Reject oversized raw axes before any graph or result model copying.
    place_count, transition_count, label_characters, parented_markings = (
        _preflight_terminal_scc_graph_shape(source_graph)
    )
    state_count = len(source_graph.states)
    edge_count = len(source_graph.edges)
    work = (
        state_count * max(1, transition_count) * max(1, place_count)
        + edge_count * max(1, place_count)
        + state_count
        + edge_count
        + (parented_markings + 1) * place_count * transition_count
        + transition_count * (state_count + edge_count)
        + label_characters
    )
    if work > MAX_TRANSITION_LIVENESS_WORK:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.transition_liveness.work_bound",
            message="transition-liveness validation exceeds its admitted work bound",
        )
    if work > MAX_TERMINAL_SCC_PROFILE_WORK:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.transition_liveness.work_bound",
            message="transition-liveness validation exceeds its graph-admission bound",
        )
    if not source_graph.truncated:
        _validate_terminal_scc_net_values(source_graph)
    output_bound = _output_size(source_graph, transition_count)
    if output_bound > min(
        MAX_TRANSITION_LIVENESS_OUTPUT_BYTES,
        MAX_TERMINAL_SCC_PROFILE_OUTPUT_BYTES,
    ):
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.transition_liveness.output_bound",
            message="transition-liveness profile exceeds its serialized output bound",
        )

    try:
        graph = ReachabilityResult.model_validate(
            source_graph.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("source_graph",),
            code="petri_net.transition_liveness.graph_shape",
            message="source graph must satisfy its bounded state and transition axes",
        ) from exc
    if (
        state_count == 0
        or state_count > graph.max_states
        or graph.states[0].marking.tokens != graph.initial_marking.tokens
    ):
        raise OperationDomainValidationError(
            location=("source_graph", "states"),
            code="petri_net.transition_liveness.state_axis",
            message="source graph must begin at its initial marking and fit its state limit",
        )
    state_indices: dict[tuple[int, ...], int] = {}
    for state in graph.states:
        tokens = state.marking.tokens
        if tokens in state_indices:
            raise OperationDomainValidationError(
                location=("source_graph", "states"),
                code="petri_net.transition_liveness.duplicate_state",
                message="source graph must contain each marking exactly once",
            )
        state_indices[tokens] = state.state_index
    if graph.truncated:
        return TransitionLivenessResult.model_construct(
            source_graph=graph,
            transitions=tuple(
                TransitionLivenessEntry.model_construct(
                    transition=transition, status="UNKNOWN", witness_state=None
                )
                for transition in range(transition_count)
            ),
        )
    _, reverse = _terminal_scc_adjacency(graph, graph.net, state_indices)
    return TransitionLivenessResult.model_construct(
        source_graph=graph,
        transitions=_classify_transitions(graph, reverse, transition_count),
    )
