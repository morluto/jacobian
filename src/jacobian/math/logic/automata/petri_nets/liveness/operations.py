"""Kernels for Petri-net transition liveness profiles."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_TERMINAL_SCC_PROFILE_WORK,
    ReachabilityResult,
)
from jacobian.math.logic.automata.petri_nets.liveness._models import (
    MAX_TRANSITION_LIVENESS_OUTPUT_CELLS,
    MAX_TRANSITION_LIVENESS_WORK,
    TransitionLivenessEntry,
    TransitionLivenessResult,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    _preflight_terminal_scc_graph_shape,
    _terminal_scc_adjacency,
    _validate_terminal_scc_net_values,
)


def _output_size(graph: ReachabilityResult, transition_count: int) -> int:
    """Count the retained profile entries, not their encoded size.

    The profile retains the net, both marking axes, every represented state's
    place axis and marking, and every reachability edge, so the cell count is
    the number of those entries. Scalar magnitudes stay exact and unbounded;
    only the number of retained entries is admitted.
    """
    net = graph.net
    cells = 2 * net.place_count + 2 * net.transition_count
    for axis in (net.place_ids, net.transition_ids):
        if axis is not None:
            cells += len(axis)
    for state in graph.states:
        cells += 1 + len(state.place_axis) + len(state.marking.tokens)
    return cells + len(graph.edges) + transition_count


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
        # The preflight's fourth value already accumulates the total matrix-cell
        # work of every parent-bound marking, so add it directly. Multiplying it
        # by the source dimensions again squared that cost and refused graphs
        # that need only bounded validation, while a degenerate empty source
        # axis zeroed it before the forged parent nets were copied.
        + parented_markings
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
    if output_bound > MAX_TRANSITION_LIVENESS_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.transition_liveness.output_bound",
            message="transition-liveness profile exceeds its admitted output cell bound",
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
