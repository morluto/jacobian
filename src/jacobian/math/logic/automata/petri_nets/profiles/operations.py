"""Kernels for exact finite Petri-net reachability token profiles."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.liveness.operations import (
    _output_size as _liveness_output_size,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    _preflight_terminal_scc_graph_shape,
    _terminal_scc_adjacency,
    _validate_terminal_scc_net_values,
)
from jacobian.math.logic.automata.petri_nets.profiles._models import (
    MAX_REACHABILITY_TOKEN_PROFILE_OUTPUT_BYTES,
    MAX_REACHABILITY_TOKEN_PROFILE_WORK,
    ReachabilityTokenProfileResult,
    ReachabilityTokenRange,
)


def reachability_token_profile(
    source_graph: ReachabilityResult,
) -> ReachabilityTokenProfileResult:
    """Compute exact token extrema across all represented reachable markings.

    Extrema describe the full reachable marking set only when ``source_graph``
    is closed. For a truncated graph they describe its represented prefix only.
    """
    places, transitions, _, parented_markings = _preflight_terminal_scc_graph_shape(
        source_graph
    )
    states = len(source_graph.states)
    edges = len(source_graph.edges)
    work = (
        states * max(1, places)
        + edges * max(1, places)
        + states * max(1, transitions) * max(1, places)
        + states
        + edges
        + parented_markings * places * transitions
    )
    if work > MAX_REACHABILITY_TOKEN_PROFILE_WORK:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.reachability_token_profile.work_bound",
            message="token-profile validation exceeds its admitted work bound",
        )
    # The profile embeds the graph so its axes and finite-state context remain
    # available to downstream operations. Reuse the conservative graph-plus-
    # transition-profile bound and add a fixed maximum-place profile allowance.
    output_bound = _liveness_output_size(source_graph, transitions) + places * 128
    if output_bound > MAX_REACHABILITY_TOKEN_PROFILE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("source_graph",),
            code="petri_net.reachability_token_profile.output_bound",
            message="token profile exceeds its serialized output bound",
        )
    _validate_terminal_scc_net_values(source_graph)
    try:
        graph = ReachabilityResult.model_validate(
            source_graph.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("source_graph",),
            code="petri_net.reachability_token_profile.graph_shape",
            message="source graph must satisfy its declared axes and bounds",
        ) from exc
    if graph.states[0].marking.tokens != graph.initial_marking.tokens:
        raise OperationDomainValidationError(
            location=("source_graph", "states"),
            code="petri_net.reachability_token_profile.initial_state",
            message="state zero must be the graph's initial marking",
        )
    indices: dict[tuple[int, ...], int] = {}
    for state in graph.states:
        tokens = state.marking.tokens
        if tokens in indices:
            raise OperationDomainValidationError(
                location=("source_graph", "states"),
                code="petri_net.reachability_token_profile.duplicate_state",
                message="source graph must contain every marking at most once",
            )
        indices[tokens] = state.state_index
    _terminal_scc_adjacency(graph, graph.net, indices)

    first_tokens = graph.states[0].marking.tokens
    minima = list(first_tokens)
    maxima = list(first_tokens)
    minimum_states = [0] * places
    maximum_states = [0] * places
    sums = [sum(first_tokens)]
    minimum_sum_state = maximum_sum_state = 0
    for state in graph.states[1:]:
        tokens = state.marking.tokens
        total = sum(tokens)
        sums.append(total)
        if total < sums[minimum_sum_state]:
            minimum_sum_state = state.state_index
        if total > sums[maximum_sum_state]:
            maximum_sum_state = state.state_index
        for place, token in enumerate(tokens):
            if token < minima[place]:
                minima[place] = token
                minimum_states[place] = state.state_index
            if token > maxima[place]:
                maxima[place] = token
                maximum_states[place] = state.state_index
    place_ranges = tuple(
        ReachabilityTokenRange.model_construct(
            place=place,
            minimum=minima[place],
            minimum_state=minimum_states[place],
            maximum=maxima[place],
            maximum_state=maximum_states[place],
        )
        for place in range(places)
    )
    min_sum, max_sum = min(sums), max(sums)
    return ReachabilityTokenProfileResult._from_kernel(
        source_graph=graph,
        completeness="OBSERVED_PREFIX" if graph.truncated else "COMPLETE",
        place_ranges=place_ranges,
        total_minimum=min_sum,
        total_minimum_state=minimum_sum_state,
        total_maximum=max_sum,
        total_maximum_state=maximum_sum_state,
    )
