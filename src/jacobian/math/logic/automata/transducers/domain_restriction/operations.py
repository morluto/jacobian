"""Exact bounded restriction of subsequential functions to regular domains."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NoReturn

from pydantic import ValidationError

from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers.domain_restriction._models import (
    SubsequentialDomainRestrictionRequest,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_ALPHABET,
    MAX_FST_EDGES,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular.values import MAX_DFA_STATES

MAX_DOMAIN_RESTRICTION_PRODUCT_STATES = MAX_FST_STATES * MAX_DFA_STATES
MAX_DOMAIN_RESTRICTION_PRODUCT_TRANSITIONS = (
    MAX_DOMAIN_RESTRICTION_PRODUCT_STATES * MAX_FST_ALPHABET
)
MAX_DOMAIN_RESTRICTION_WORK = 8_000_000

_ProductPair = tuple[int, int]
_ProductEdge = tuple[int, int, int, tuple[int, ...]]


@dataclass(slots=True)
class _ProductGraph:
    states: list[_ProductPair]
    edges: list[_ProductEdge]
    reverse_edges: list[list[int]]
    final_states: set[int]


def _reject(code: str, message: str, *location: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_state_transducer.{code}",
        message=message,
    )


def _admit_request(
    request: object,
) -> SubsequentialDomainRestrictionRequest:
    if not isinstance(request, SubsequentialDomainRestrictionRequest):
        _reject(
            "domain_restriction_request_type",
            "request must be a SubsequentialDomainRestrictionRequest",
        )
    try:
        return SubsequentialDomainRestrictionRequest.model_validate(
            request.model_dump(), strict=True
        )
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="finite_state_transducer.domain_restriction_carrier_shape",
            message=(
                "transducer and DFA must satisfy their canonical carrier shapes "
                "and share an input alphabet"
            ),
        ) from exc


def _preflight(
    request: SubsequentialDomainRestrictionRequest,
) -> tuple[OperationWorkLedger, int]:
    transducer = request.transducer
    dfa = request.domain_dfa
    source_label_cells = sum(len(edge.output) for edge in transducer.transitions)
    final_label_cells = sum(len(final.output) for final in transducer.final_outputs)
    product_state_bound = transducer.state_count * dfa.state_count
    product_transition_bound = product_state_bound * transducer.input_alphabet_size
    # The canonical result carrier bounds the labels duplicated by the product.
    result_label_cells_bound = (
        min(MAX_FST_EDGES, product_transition_bound)
        + min(MAX_FST_STATES, product_state_bound)
    ) * MAX_FST_WORD_LENGTH
    result_record_bound = 2 * (MAX_FST_EDGES + MAX_FST_STATES)
    work_bound = (
        len(transducer.transitions)
        + source_label_cells
        + len(transducer.final_outputs)
        + final_label_cells
        + len(dfa.transitions)
        + 3 * product_transition_bound
        + 2 * result_label_cells_bound
        + result_record_bound
    )
    if (
        product_state_bound > MAX_DOMAIN_RESTRICTION_PRODUCT_STATES
        or product_transition_bound > MAX_DOMAIN_RESTRICTION_PRODUCT_TRANSITIONS
        or work_bound > MAX_DOMAIN_RESTRICTION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("transducer", "domain_dfa"),
            code="finite_state_transducer.domain_restriction_work_bound_exceeded",
            message="reachable transducer-DFA product exceeds the admitted work bound",
        )
    request_checkpoint("after subsequential domain restriction admission")
    return OperationWorkLedger(work_bound), result_label_cells_bound


def _explore_product(
    request: SubsequentialDomainRestrictionRequest,
    ledger: OperationWorkLedger,
) -> _ProductGraph:
    transducer = request.transducer
    dfa = request.domain_dfa
    transducer_transitions = {
        (edge.source, edge.input_symbol): edge for edge in transducer.transitions
    }
    dfa_transitions = {
        (edge.source, edge.symbol): edge.target for edge in dfa.transitions
    }
    final_outputs = {item.state: item.output for item in transducer.final_outputs}
    accepting = set(dfa.accepting_states)

    initial = (transducer.initial_state, dfa.initial_state)
    states = [initial]
    state_ids = {initial: 0}
    edges: list[_ProductEdge] = []
    reverse_edges: list[list[int]] = [[]]
    final_states: set[int] = set()
    cursor = 0
    while cursor < len(states):
        request_checkpoint("during subsequential domain restriction product search")
        transducer_state, dfa_state = states[cursor]
        if transducer_state in final_outputs and dfa_state in accepting:
            final_states.add(cursor)
        for symbol in range(transducer.input_alphabet_size):
            ledger.charge()
            edge = transducer_transitions.get((transducer_state, symbol))
            if edge is None:
                continue
            target_pair = (edge.target, dfa_transitions[(dfa_state, symbol)])
            target_id = state_ids.get(target_pair)
            if target_id is None:
                target_id = len(states)
                state_ids[target_pair] = target_id
                states.append(target_pair)
                reverse_edges.append([])
            edges.append((cursor, symbol, target_id, edge.output))
            reverse_edges[target_id].append(cursor)
        cursor += 1
    return _ProductGraph(states, edges, reverse_edges, final_states)


def _coaccessible_states(graph: _ProductGraph, ledger: OperationWorkLedger) -> set[int]:
    coaccessible = set(graph.final_states)
    frontier = deque(graph.final_states)
    while frontier:
        request_checkpoint("during subsequential domain restriction trim")
        target = frontier.popleft()
        for source in graph.reverse_edges[target]:
            ledger.charge()
            if source not in coaccessible:
                coaccessible.add(source)
                frontier.append(source)
    return coaccessible


def _empty_function(transducer: SubsequentialTransducer) -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=transducer.input_alphabet_size,
        output_alphabet_size=transducer.output_alphabet_size,
        input_alphabet_id=transducer.input_alphabet_id,
        output_alphabet_id=transducer.output_alphabet_id,
        input_alphabet=transducer.input_alphabet,
        output_alphabet=transducer.output_alphabet,
        state_count=1,
        initial_state=0,
        transitions=(),
        final_outputs=(),
    )


def _build_restricted_transducer(
    request: SubsequentialDomainRestrictionRequest,
    graph: _ProductGraph,
    coaccessible: set[int],
    ledger: OperationWorkLedger,
    result_label_cells_bound: int,
) -> SubsequentialTransducer:
    transducer = request.transducer
    if 0 not in coaccessible:
        return _empty_function(transducer)

    kept_states = tuple(
        state_id for state_id in range(len(graph.states)) if state_id in coaccessible
    )
    if len(kept_states) > MAX_FST_STATES:
        raise OperationResourceAdmissionError(
            location=("transducer", "domain_dfa"),
            code="finite_state_transducer.domain_restriction_state_bound_exceeded",
            message="trimmed product has more states than the canonical transducer carrier",
        )
    compact_ids = {old: new for new, old in enumerate(kept_states)}

    retained_edges: list[_ProductEdge] = []
    for edge in graph.edges:
        ledger.charge()
        if edge[0] in coaccessible and edge[2] in coaccessible:
            retained_edges.append(edge)
    if len(retained_edges) > MAX_FST_EDGES:
        raise OperationResourceAdmissionError(
            location=("transducer", "domain_dfa"),
            code="finite_state_transducer.domain_restriction_transition_bound_exceeded",
            message="trimmed product exceeds the canonical transducer transition limit",
        )
    retained_finals = tuple(
        state for state in kept_states if state in graph.final_states
    )
    source_finals = {item.state: item.output for item in transducer.final_outputs}
    output_label_cells = sum(len(edge[3]) for edge in retained_edges) + sum(
        len(source_finals[graph.states[state][0]]) for state in retained_finals
    )
    if output_label_cells > result_label_cells_bound:
        raise OperationResourceAdmissionError(
            location=("transducer", "domain_dfa"),
            code="finite_state_transducer.domain_restriction_output_bound_exceeded",
            message="restricted transducer labels exceed the admitted result bound",
        )

    transitions: list[SubseqTransition] = []
    for source, symbol, target, output in retained_edges:
        ledger.charge(len(output) + 1)
        transitions.append(
            SubseqTransition(
                source=compact_ids[source],
                input_symbol=symbol,
                target=compact_ids[target],
                output=output,
            )
        )
    finals = tuple(
        SubseqFinalOutput(
            state=compact_ids[state],
            output=source_finals[graph.states[state][0]],
        )
        for state in retained_finals
    )
    ledger.charge(sum(len(final.output) + 1 for final in finals))
    request_checkpoint("before subsequential domain restriction result construction")
    return SubsequentialTransducer(
        input_alphabet_size=transducer.input_alphabet_size,
        output_alphabet_size=transducer.output_alphabet_size,
        input_alphabet_id=transducer.input_alphabet_id,
        output_alphabet_id=transducer.output_alphabet_id,
        input_alphabet=transducer.input_alphabet,
        output_alphabet=transducer.output_alphabet,
        state_count=len(kept_states),
        initial_state=compact_ids[0],
        transitions=tuple(transitions),
        final_outputs=finals,
    )


def restrict_subsequential_domain(
    request: SubsequentialDomainRestrictionRequest,
) -> SubsequentialTransducer:
    """Return the same partial function restricted to the supplied DFA language."""

    request = _admit_request(request)
    ledger, label_bound = _preflight(request)
    graph = _explore_product(request, ledger)
    coaccessible = _coaccessible_states(graph, ledger)
    return _build_restricted_transducer(
        request, graph, coaccessible, ledger, label_bound
    )


__all__ = ["restrict_subsequential_domain"]
