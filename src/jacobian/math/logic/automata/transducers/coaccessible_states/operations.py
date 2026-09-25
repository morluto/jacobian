"""Exact bounded successful-continuation witnesses."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers.coaccessible_states._models import (
    CoaccessibleStatesRequest,
    CoaccessibleStateWitness,
    CoaccessibleStateWitnesses,
)
from jacobian.math.logic.automata.transducers.operations import _admit_transducer
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    SubseqTransition,
    SubsequentialTransducer,
)

MAX_SOURCE_BYTES = 8_000_000
MAX_RESULT_BYTES = 8_500_000


def _resource_error(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("transducer",),
        code=f"finite_state_transducer.{code}",
        message=message,
    )


def _result_byte_bound(
    source_bytes: int,
    paths: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    output_lengths: tuple[int, ...],
) -> int:
    # Every symbol is an integer in 0..31, transition indices are at most
    # 4095, and JSON arrays need at most one comma per entry. This deliberately
    # includes a generous fixed row overhead for field names and endpoints.
    return source_bytes + sum(
        256 + 3 * len(suffix) + 3 * len(trace) + 6 * len(suffix) + 3 * output_length
        for (_state, suffix, trace), output_length in zip(
            paths, output_lengths, strict=True
        )
    )


def coaccessible_state_witnesses(
    transducer: SubsequentialTransducer,
) -> CoaccessibleStateWitnesses:
    """Return a shortest successful suffix and its exact output for each state.

    Equal-length suffixes are ordered lexicographically by input symbol. A
    state is coaccessible precisely when some suffix reaches a state with a
    final output. Output for each witness is the transition outputs in path
    order followed by that terminal final output.
    """

    request = CoaccessibleStatesRequest(transducer=transducer)
    source_bytes = len(request.transducer.model_dump_json().encode("utf-8"))
    if source_bytes > MAX_SOURCE_BYTES:
        raise _resource_error(
            "coaccessible_source_bytes_exceeded",
            f"transducer serialization exceeds {MAX_SOURCE_BYTES} bytes",
        )
    value = _admit_transducer(request.transducer)

    outgoing: list[list[tuple[int, int, SubseqTransition]]] = [
        [] for _ in range(value.state_count)
    ]
    transition_indices: dict[tuple[int, int], int] = {}
    finals = {item.state: item.output for item in value.final_outputs}
    reverse: list[list[int]] = [[] for _ in range(value.state_count)]
    for index, edge in enumerate(value.transitions):
        outgoing[edge.source].append((edge.input_symbol, index, edge))
        transition_indices[(edge.source, edge.input_symbol)] = index
        reverse[edge.target].append(edge.source)

    distance: dict[int, int] = dict.fromkeys(finals, 0)
    queue = deque(sorted(finals))
    while queue:
        target = queue.popleft()
        for source in reverse[target]:
            if source not in distance:
                distance[source] = distance[target] + 1
                queue.append(source)

    choices = {
        state: min(
            edge_row
            for edge_row in outgoing[state]
            if distance.get(edge_row[2].target) == distance[state] - 1
        )
        for state in distance
        if distance[state] > 0
    }

    paths: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    output_parts: list[tuple[tuple[int, ...], ...]] = []
    output_lengths: list[int] = []
    for start in sorted(distance):
        state = start
        suffix: list[int] = []
        trace = [state]
        parts: list[tuple[int, ...]] = []
        while distance[state] > 0:
            next_edge = choices[state]
            symbol, _index, edge = next_edge
            suffix.append(symbol)
            parts.append(edge.output)
            state = edge.target
            trace.append(state)
        terminal_output = finals[state]
        parts.append(terminal_output)
        output_length = sum(map(len, parts))
        if output_length > MAX_FST_RESULT_WORD_LENGTH:
            raise _resource_error(
                "coaccessible_witness_output_exceeded",
                "a successful continuation output exceeds the result word bound",
            )
        paths.append((start, tuple(suffix), tuple(trace)))
        output_parts.append(tuple(parts))
        output_lengths.append(output_length)

    path_data = tuple(paths)
    edge_paths = tuple(
        tuple(transition_indices[(trace[i], suffix[i])] for i in range(len(suffix)))
        for _state, suffix, trace in path_data
    )
    result_bytes = _result_byte_bound(
        source_bytes,
        path_data,
        tuple(output_lengths),
    )
    if result_bytes > MAX_RESULT_BYTES:
        raise _resource_error(
            "coaccessible_result_bytes_exceeded",
            f"coaccessible result may exceed {MAX_RESULT_BYTES} bytes",
        )

    witnesses = tuple(
        CoaccessibleStateWitness(
            state=state,
            input_suffix=suffix,
            state_trace=trace,
            transition_indices=edge_path,
            final_state=trace[-1],
            output_word=tuple(symbol for part in parts for symbol in part),
        )
        for (state, suffix, trace), edge_path, parts in zip(
            path_data, edge_paths, output_parts, strict=True
        )
    )
    return CoaccessibleStateWitnesses.model_construct(
        transducer=value,
        witnesses=witnesses,
    )


__all__ = ["coaccessible_state_witnesses"]
