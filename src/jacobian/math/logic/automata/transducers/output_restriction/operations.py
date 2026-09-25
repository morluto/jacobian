"""Exact bounded output-tape restriction for rational transducers."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    MAX_RESTRICT_OUTPUT_LABEL_CELLS,
    MAX_RESTRICT_OUTPUT_RESULT_EDGES,
    MAX_RESTRICT_OUTPUT_WORK,
    RestrictOutputEdgeSource,
    RestrictOutputProductState,
    RestrictRationalOutputRequest,
    RestrictRationalOutputResult,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_ALPHABET,
    MAX_FST_EDGES,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_DFA_ALPHABET,
    MAX_DFA_STATES,
    MAX_DFA_TRANSITIONS,
)


def _fail(code: str, message: str, *, resource: bool = False) -> None:
    error = (
        OperationResourceAdmissionError if resource else OperationDomainValidationError
    )
    raise error(
        location=("transducer",),
        code=f"rational_transducer.restrict_output.{code}",
        message=message,
    )


def _utf8_length(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return MAX_RESTRICT_OUTPUT_LABEL_CELLS + 1


def _validate_relation_axes(relation: RationalTransducer) -> None:
    if (
        type(relation.state_count) is not int
        or not 1 <= relation.state_count <= MAX_FST_STATES
    ):
        _fail(
            "state_count_invalid", "transducer state count is outside its carrier bound"
        )
    if (
        type(relation.input_alphabet_size) is not int
        or type(relation.output_alphabet_size) is not int
        or not 1 <= relation.input_alphabet_size <= MAX_FST_ALPHABET
        or not 1 <= relation.output_alphabet_size <= MAX_FST_ALPHABET
    ):
        _fail(
            "alphabet_size_invalid",
            "transducer alphabet size is outside its carrier bound",
        )
    if not isinstance(relation.edges, tuple) or len(relation.edges) > MAX_FST_EDGES:
        _fail(
            "edge_count_exceeded",
            "transducer edge count exceeds its carrier bound",
            resource=True,
        )
    if not isinstance(relation.initial_states, tuple) or not relation.initial_states:
        _fail(
            "initial_states_invalid",
            "transducer initial states must be a nonempty tuple",
        )
    if not isinstance(relation.accepting_states, tuple):
        _fail("accepting_states_invalid", "transducer accepting states must be a tuple")
    states = (*relation.initial_states, *relation.accepting_states)
    if any(
        type(state) is not int or not 0 <= state < relation.state_count
        for state in states
    ):
        _fail(
            "state_out_of_range",
            "transducer initial and accepting states must be declared",
        )
    if len(set(relation.initial_states)) != len(relation.initial_states):
        _fail("initial_states_invalid", "transducer initial states must be distinct")
    if len(set(relation.accepting_states)) != len(relation.accepting_states):
        _fail(
            "accepting_states_invalid", "transducer accepting states must be distinct"
        )


def _validate_alphabet_contexts(
    relation: RationalTransducer, alphabet: FiniteAlphabet
) -> None:
    if not isinstance(alphabet, FiniteAlphabet) or not isinstance(
        alphabet.symbols, tuple
    ):
        _fail(
            "output_context_invalid", "output context must be a finite alphabet value"
        )
    if len(alphabet.symbols) != relation.output_alphabet_size:
        _fail(
            "output_context_size_mismatch",
            "output alphabet context must match the output axis",
        )
    for context in (relation.input_alphabet, relation.output_alphabet, alphabet):
        if context is None:
            continue
        if not isinstance(context, FiniteAlphabet) or not isinstance(
            context.symbols, tuple
        ):
            _fail(
                "alphabet_context_invalid",
                "alphabet contexts must be finite alphabet values",
            )
        if len(context.symbols) > MAX_FST_ALPHABET or any(
            not isinstance(symbol, str) or _utf8_length(symbol) > 256
            for symbol in context.symbols
        ):
            _fail(
                "alphabet_symbol_too_long",
                "alphabet labels must be at most 256 UTF-8 bytes",
            )
    if any(
        identifier is not None
        and (not isinstance(identifier, str) or _utf8_length(identifier) > 256)
        for identifier in (relation.input_alphabet_id, relation.output_alphabet_id)
    ):
        _fail(
            "alphabet_identity_too_long",
            "alphabet identities must be at most 256 UTF-8 bytes",
        )
    if relation.output_alphabet is not None and relation.output_alphabet != alphabet:
        _fail(
            "output_context_mismatch",
            "output alphabet context differs from the relation",
        )


def _validate_dfa(language: DFA, output_size: int) -> dict[tuple[int, int], int]:
    if not isinstance(language, DFA):
        _fail("language_type_invalid", "output restriction requires a total DFA")
    if (
        type(language.state_count) is not int
        or not 1 <= language.state_count <= MAX_DFA_STATES
    ):
        _fail("dfa_state_count_invalid", "DFA state count is outside its carrier bound")
    if type(language.alphabet_size) is not int or language.alphabet_size != output_size:
        _fail(
            "alphabet_size_mismatch",
            "DFA alphabet size must match the transducer output alphabet",
        )
    if (
        not isinstance(language.transitions, tuple)
        or len(language.transitions) > MAX_DFA_TRANSITIONS
        or language.alphabet_size > MAX_DFA_ALPHABET
    ):
        _fail("dfa_size_exceeded", "DFA size exceeds its carrier bound", resource=True)
    if (
        type(language.initial_state) is not int
        or not 0 <= language.initial_state < language.state_count
    ):
        _fail("dfa_initial_state_invalid", "DFA initial state must be declared")
    if (
        not isinstance(language.accepting_states, tuple)
        or len(language.accepting_states) > MAX_DFA_STATES
    ):
        _fail(
            "dfa_accepting_states_invalid",
            "DFA accepting states must be a bounded tuple",
        )
    accepting = set(language.accepting_states)
    if len(accepting) != len(language.accepting_states) or any(
        type(state) is not int or not 0 <= state < language.state_count
        for state in accepting
    ):
        _fail(
            "dfa_accepting_states_invalid",
            "DFA accepting states must be distinct and declared",
        )
    delta: dict[tuple[int, int], int] = {}
    for transition in language.transitions:
        if not all(
            hasattr(transition, field) for field in ("source", "symbol", "target")
        ):
            _fail(
                "dfa_transition_invalid",
                "DFA transition rows must be typed transitions",
            )
        if not (
            type(transition.source) is int
            and type(transition.target) is int
            and type(transition.symbol) is int
            and 0 <= transition.source < language.state_count
            and 0 <= transition.target < language.state_count
            and 0 <= transition.symbol < language.alphabet_size
        ):
            _fail("dfa_transition_invalid", "DFA transition leaves its declared axes")
        key = (transition.source, transition.symbol)
        if key in delta:
            _fail(
                "dfa_transition_duplicate", "DFA has duplicate state-symbol transitions"
            )
        delta[key] = transition.target
    if len(delta) != language.state_count * language.alphabet_size:
        _fail("dfa_not_total", "output language must be a total DFA")
    return delta


def _validate_relation_edges(
    relation: RationalTransducer,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    int,
]:
    outgoing: list[list[int]] = [[] for _ in range(relation.state_count)]
    edge_inputs: list[tuple[int, ...]] = []
    edge_outputs: list[tuple[int, ...]] = []
    label_cells = 0
    for edge_index, edge in enumerate(relation.edges):
        if not all(
            hasattr(edge, field)
            for field in ("source", "target", "input_label", "output_label")
        ):
            _fail("edge_invalid", "transducer rows must be typed rational edges")
        if (
            type(edge.source) is not int
            or type(edge.target) is not int
            or not (
                0 <= edge.source < relation.state_count
                and 0 <= edge.target < relation.state_count
            )
        ):
            _fail("edge_state_invalid", "transducer edge leaves its state domain")
        if not isinstance(edge.input_label, tuple) or not isinstance(
            edge.output_label, tuple
        ):
            _fail(
                "edge_label_invalid",
                "transducer edge labels must be finite word tuples",
            )
        if not edge.input_label and not edge.output_label:
            _fail("edge_labels_empty", "transducer edges cannot be empty on both tapes")
        if (
            len(edge.input_label) > MAX_FST_WORD_LENGTH
            or len(edge.output_label) > MAX_FST_WORD_LENGTH
        ):
            _fail(
                "edge_label_too_long",
                "transducer edge label exceeds its carrier bound",
                resource=True,
            )
        if any(
            type(symbol) is not int or not 0 <= symbol < relation.input_alphabet_size
            for symbol in edge.input_label
        ):
            _fail("input_label_invalid", "transducer input label leaves its alphabet")
        if any(
            type(symbol) is not int or not 0 <= symbol < relation.output_alphabet_size
            for symbol in edge.output_label
        ):
            _fail("output_label_invalid", "transducer output label leaves its alphabet")
        label_cells += len(edge.input_label) + len(edge.output_label)
        outgoing[edge.source].append(edge_index)
        edge_inputs.append(edge.input_label)
        edge_outputs.append(edge.output_label)
    return (
        tuple(tuple(row) for row in outgoing),
        tuple(edge_inputs),
        tuple(edge_outputs),
        label_cells,
    )


def _validate_inputs(
    request: RestrictRationalOutputRequest,
) -> tuple[
    dict[tuple[int, int], int], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
]:
    relation = request.transducer
    if not isinstance(relation, RationalTransducer):
        _fail(
            "relation_type_invalid", "output restriction requires a rational transducer"
        )
    _validate_relation_axes(relation)
    if not isinstance(request.output_language, DFA):
        _fail("language_type_invalid", "output restriction requires a total DFA")
    delta = _validate_dfa(request.output_language, relation.output_alphabet_size)
    _validate_alphabet_contexts(relation, request.output_alphabet)
    outgoing, edge_inputs, edge_outputs, label_cells = _validate_relation_edges(
        relation
    )
    if label_cells > MAX_RESTRICT_OUTPUT_LABEL_CELLS:
        _fail(
            "label_cells_exceeded",
            "aggregate transducer label size exceeds output-restriction admission",
            resource=True,
        )
    return delta, outgoing, (edge_inputs, edge_outputs)


def restrict_rational_output(
    request: RestrictRationalOutputRequest,
) -> RestrictRationalOutputResult:
    """Return the exact relation ``R ∩ (A* x L)``.

    Product construction advances the DFA over each whole output-edge label.
    The admitted planning pass records only bounded reachable product states
    and source-edge coordinates; canonical target edges are materialized after
    exact state, edge, label-cell, and output bounds are known.
    """
    if not isinstance(request, RestrictRationalOutputRequest):
        _fail(
            "request_type_invalid", "output restriction requires its canonical request"
        )
    delta, outgoing, labels = _validate_inputs(request)
    relation = request.transducer
    language = request.output_language
    edge_inputs, edge_outputs = labels

    product_state_bound = relation.state_count * language.state_count
    max_output_label = max((len(label) for label in edge_outputs), default=0)
    work_bound = (
        len(relation.edges)
        + len(language.transitions)
        + sum(
            len(edge_input) + len(edge_output)
            for edge_input, edge_output in zip(edge_inputs, edge_outputs, strict=True)
        )
        + product_state_bound * max(1, len(relation.edges)) * max(1, max_output_label)
    )
    if work_bound > MAX_RESTRICT_OUTPUT_WORK:
        _fail(
            "work_bound_exceeded",
            f"worst-case product work {work_bound} exceeds {MAX_RESTRICT_OUTPUT_WORK}",
            resource=True,
        )

    # The plan pass is itself bounded above by work_bound. It computes exact
    # reachable output size before constructing any RationalEdge output rows.
    initial_pairs = sorted(
        (state, language.initial_state) for state in relation.initial_states
    )
    pair_to_id: dict[tuple[int, int], int] = {
        pair: index for index, pair in enumerate(initial_pairs)
    }
    if len(initial_pairs) > MAX_FST_STATES:
        _fail(
            "product_states_exceeded",
            "initial product states exceed the output carrier",
            resource=True,
        )
    pairs = list(initial_pairs)
    queue = deque(initial_pairs)
    arc_plan: list[tuple[int, int, int]] = []
    while queue:
        pair = queue.popleft()
        source_id = pair_to_id[pair]
        relation_state, dfa_state = pair
        for edge_index in outgoing[relation_state]:
            next_dfa_state = dfa_state
            for symbol in edge_outputs[edge_index]:
                next_dfa_state = delta[(next_dfa_state, symbol)]
            edge = relation.edges[edge_index]
            target_pair = (edge.target, next_dfa_state)
            target_id = pair_to_id.get(target_pair)
            if target_id is None:
                if len(pairs) >= relation.state_count * language.state_count:
                    _fail(
                        "product_state_accounting_failed",
                        "reachable product states exceed their admitted bound",
                    )
                target_id = len(pairs)
                pair_to_id[target_pair] = target_id
                pairs.append(target_pair)
                queue.append(target_pair)
            if len(arc_plan) >= MAX_RESTRICT_OUTPUT_RESULT_EDGES:
                _fail(
                    "product_edges_exceeded",
                    "reachable product relation exceeds the 4,096-edge output bound",
                    resource=True,
                )
            arc_plan.append((source_id, edge_index, target_id))

    if len(pairs) > MAX_FST_STATES:
        _fail(
            "product_states_exceeded",
            "reachable product states exceed the 64-state output bound",
            resource=True,
        )
    output_label_cells = sum(
        len(edge_inputs[edge_index]) + len(edge_outputs[edge_index])
        for _, edge_index, _ in arc_plan
    )
    if output_label_cells > MAX_RESTRICT_OUTPUT_LABEL_CELLS:
        _fail(
            "output_label_cells_exceeded",
            "restricted relation labels exceed output admission",
            resource=True,
        )

    product_states = tuple(
        RestrictOutputProductState(
            product_state=product_state,
            transducer_state=source_state,
            language_state=dfa_state,
        )
        for product_state, (source_state, dfa_state) in enumerate(pairs)
    )
    initial_ids = tuple(pair_to_id[pair] for pair in initial_pairs)
    accepting_source_states = set(relation.accepting_states)
    accepting_language_states = set(language.accepting_states)
    accepting_ids = tuple(
        product_state
        for product_state, (source_state, dfa_state) in enumerate(pairs)
        if source_state in accepting_source_states
        and dfa_state in accepting_language_states
    )
    restricted_edges = tuple(
        RationalEdge(
            source=product_source,
            target=product_target,
            input_label=edge_inputs[source_edge],
            output_label=edge_outputs[source_edge],
        )
        for product_source, source_edge, product_target in arc_plan
    )
    restricted = RationalTransducer(
        input_alphabet_size=relation.input_alphabet_size,
        output_alphabet_size=relation.output_alphabet_size,
        input_alphabet_id=relation.input_alphabet_id,
        output_alphabet_id=relation.output_alphabet_id,
        input_alphabet=relation.input_alphabet,
        output_alphabet=request.output_alphabet,
        state_count=len(pairs),
        initial_states=initial_ids,
        accepting_states=accepting_ids,
        edges=restricted_edges,
    )
    edge_sources = tuple(
        RestrictOutputEdgeSource(restricted_edge=index, source_edge=source_edge)
        for index, (_, source_edge, _) in enumerate(arc_plan)
    )
    return RestrictRationalOutputResult(
        source=relation,
        output_language=language,
        output_alphabet=request.output_alphabet,
        restricted=restricted,
        product_states=product_states,
        edge_sources=edge_sources,
    )


__all__ = ["restrict_rational_output"]
