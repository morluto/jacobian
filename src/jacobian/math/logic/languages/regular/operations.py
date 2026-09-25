"""Exact finite-automaton and regular-language kernels."""

from __future__ import annotations

from collections import deque

from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers.values import SubsequentialTransducer
from jacobian.math.logic.languages.regular._models import CountResult, RunResult
from jacobian.math.logic.languages.regular._profile_admission import (
    TransitionParikhAdmissionPlan,
    admit_transition_profile,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_MATRIX_BIT_WORK,
    MAX_COUNT_RESULT_DIGITS,
    MAX_COUNT_WORD_LENGTH,
    MAX_DFA_ALPHABET,
    MAX_DFA_ALPHABET_ID_LENGTH,
    MAX_DFA_EQUIVALENCE_INTERMEDIATE_ALLOCATION,
    MAX_DFA_EQUIVALENCE_OUTPUT_ALLOCATION,
    MAX_DFA_EQUIVALENCE_PRODUCT_STATES,
    MAX_DFA_EQUIVALENCE_PRODUCT_TRANSITIONS,
    MAX_DFA_EQUIVALENCE_TRACE_ROWS,
    MAX_DFA_EQUIVALENCE_WITNESS_LENGTH,
    MAX_DFA_EQUIVALENCE_WORK,
    MAX_DFA_PREIMAGE_OUTPUT_BYTES,
    MAX_DFA_PREIMAGE_PRODUCT_STATES,
    MAX_DFA_PREIMAGE_PRODUCT_TRANSITIONS,
    MAX_DFA_PREIMAGE_WORK,
    MAX_DFA_STATES,
    MAX_DFA_TRANSITIONS,
    MAX_NFA_MEMBERSHIP_WORK,
    MAX_NFA_STATES,
    MAX_NFA_TRANSITIONS,
    MAX_SUBSEQUENTIAL_IMAGE_INTERMEDIATE_BYTES,
    MAX_SUBSEQUENTIAL_IMAGE_OUTPUT_BYTES,
    MAX_SUBSEQUENTIAL_IMAGE_WORK,
    MAX_WORD_LENGTH,
    NFA,
    AutomatonTransition,
    DFATransition,
    FiniteAlphabet,
    FiniteLabeledAutomaton,
    NFATransition,
    TransitionParikhCell,
    TransitionParikhProfile,
)

__all__ = [
    "count_accepted_words",
    "dfa_complement",
    "dfa_equivalence",
    "dfa_run",
    "dfa_subsequential_image",
    "dfa_subsequential_preimage",
    "dfa_transition_carrier",
    "nfa_membership",
    "nfa_subsequential_image",
    "transition_parikh_profile",
    "verify_accepted_word_count",
    "verify_dfa_run",
    "verify_transition_parikh_profile",
]


def _transition_map(dfa: DFA) -> dict[tuple[int, int], int]:
    return {(tr.source, tr.symbol): tr.target for tr in dfa.transitions}


def dfa_run(dfa: DFA, word: tuple[int, ...]) -> tuple[bool, int]:
    """Simulate a total DFA on a word; return ``(accepted, final_state)``."""

    if any(not 0 <= symbol < dfa.alphabet_size for symbol in word):
        raise ValueError("word symbols must be in 0..alphabet_size-1")
    transitions = _transition_map(dfa)
    state = dfa.initial_state
    for symbol in word:
        state = transitions[(state, symbol)]
    return (state in dfa.accepting_states, state)


def nfa_membership(nfa: NFA, word: tuple[int, ...]) -> bool:
    """Decide membership by bounded state-set propagation with epsilon closure."""
    _validate_nfa_membership_input(nfa, word)
    work_bound = (2 * len(word) + 1) * (nfa.state_count + len(nfa.transitions))
    work_bound += len(nfa.transitions)
    if work_bound > MAX_NFA_MEMBERSHIP_WORK:
        raise OperationResourceAdmissionError(
            location=("nfa", "word"),
            code="regular_language.nfa_membership_work_bound",
            message="NFA membership exceeds the state-set propagation work bound",
        )
    request_checkpoint("after NFA membership admission")
    ledger = OperationWorkLedger(work_bound)
    epsilon_edges: list[list[int]] = [[] for _ in range(nfa.state_count)]
    labeled_edges: dict[tuple[int, int], list[int]] = {}
    for edge in nfa.transitions:
        ledger.charge()
        if edge.symbol is None:
            epsilon_edges[edge.source].append(edge.target)
        else:
            labeled_edges.setdefault((edge.source, edge.symbol), []).append(edge.target)
    active = _nfa_epsilon_closure({nfa.initial_state}, epsilon_edges, ledger)
    for symbol in word:
        request_checkpoint("during NFA membership propagation")
        destinations: set[int] = set()
        for state in active:
            ledger.charge()
            targets = labeled_edges.get((state, symbol), ())
            ledger.charge(len(targets))
            destinations.update(targets)
        active = _nfa_epsilon_closure(destinations, epsilon_edges, ledger)
    return not active.isdisjoint(nfa.accepting_states)


def _validate_nfa_membership_input(nfa: object, word: object) -> None:
    if type(nfa) is not NFA:
        raise OperationDomainValidationError(
            location=("nfa",),
            code="regular_language.nfa_membership.noncanonical_nfa",
            message="membership requires a canonical NFA value",
        )
    if (
        type(nfa.state_count) is not int
        or not 1 <= nfa.state_count <= MAX_NFA_STATES
        or type(nfa.alphabet_size) is not int
        or not 0 <= nfa.alphabet_size <= MAX_DFA_ALPHABET
        or type(nfa.initial_state) is not int
        or not 0 <= nfa.initial_state < nfa.state_count
        or type(nfa.alphabet) is not FiniteAlphabet
        or len(nfa.alphabet.symbols) != nfa.alphabet_size
        or (
            nfa.alphabet_id is not None
            and (
                type(nfa.alphabet_id) is not str
                or len(nfa.alphabet_id) > MAX_DFA_ALPHABET_ID_LENGTH
            )
        )
        or type(nfa.accepting_states) is not tuple
        or any(
            type(state) is not int or not 0 <= state < nfa.state_count
            for state in nfa.accepting_states
        )
        or len(set(nfa.accepting_states)) != len(nfa.accepting_states)
        or type(nfa.transitions) is not tuple
        or len(nfa.transitions) > MAX_NFA_TRANSITIONS
    ):
        raise OperationDomainValidationError(
            location=("nfa",),
            code="regular_language.nfa_membership.invalid_nfa",
            message="membership requires a valid explicitly parented NFA carrier",
        )
    if any(type(edge) is not NFATransition for edge in nfa.transitions):
        raise OperationDomainValidationError(
            location=("nfa", "transitions"),
            code="regular_language.nfa_membership.invalid_transition",
            message="NFA transitions must be canonical NFATransition values",
        )
    if tuple(edge.transition_id for edge in nfa.transitions) != tuple(
        range(len(nfa.transitions))
    ):
        raise OperationDomainValidationError(
            location=("nfa", "transitions"),
            code="regular_language.nfa_membership.invalid_transition_axis",
            message="NFA transition IDs must be the canonical ordered edge axis",
        )
    for edge in nfa.transitions:
        if (
            type(edge) is not NFATransition
            or type(edge.source) is not int
            or not 0 <= edge.source < nfa.state_count
            or type(edge.target) is not int
            or not 0 <= edge.target < nfa.state_count
            or (
                edge.symbol is not None
                and (
                    type(edge.symbol) is not int
                    or not 0 <= edge.symbol < nfa.alphabet_size
                )
            )
        ):
            raise OperationDomainValidationError(
                location=("nfa", "transitions"),
                code="regular_language.nfa_membership.invalid_transition",
                message="NFA transitions must have in-range states and symbols",
            )
    if (
        type(word) is not tuple
        or len(word) > MAX_WORD_LENGTH
        or any(
            type(symbol) is not int or not 0 <= symbol < nfa.alphabet_size
            for symbol in word
        )
    ):
        raise OperationDomainValidationError(
            location=("word",),
            code="regular_language.nfa_membership.invalid_word",
            message="word symbols must be in the NFA alphabet and within the word bound",
        )


def _nfa_epsilon_closure(
    seeds: set[int],
    epsilon_edges: list[list[int]],
    ledger: OperationWorkLedger,
) -> set[int]:
    reached = set(seeds)
    frontier = list(seeds)
    while frontier:
        state = frontier.pop()
        ledger.charge()
        for target in epsilon_edges[state]:
            ledger.charge()
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return reached


def count_accepted_words(dfa: DFA, word_length: int) -> int:
    """Count accepted words of exact length via exact integer matrix powering."""

    if type(word_length) is not int or not 0 <= word_length <= MAX_COUNT_WORD_LENGTH:
        raise OperationDomainValidationError(
            location=("word_length",),
            code="regular_language.word_length_out_of_bounds",
            message="word length is outside the exact counting bound",
        )
    if not dfa.accepting_states:
        # Empty accepting sets accept no words of any length, including ε.
        return 0
    if word_length == 0:
        return 1 if dfa.initial_state in dfa.accepting_states else 0
    accepted_paths = _accepted_path_matrix(dfa)
    if accepted_paths is None:
        return 0
    matrix, accepting_states = accepted_paths
    state_count = len(matrix)
    result_cap = 10**MAX_COUNT_RESULT_DIGITS
    max_intermediate, selected_count = _powered_count_admission(
        matrix,
        accepting_states,
        word_length,
        result_cap,
    )
    if selected_count >= result_cap:
        raise OperationResourceAdmissionError(
            location=("dfa", "word_length"),
            code="regular_language.count_result_bound",
            message="accepted-word count exceeds the canonical result digit bound",
        )
    if max_intermediate >= result_cap:
        # FLINT powers the full matrix. Unused entries can explode while the
        # selected count stays tiny; do not cap that growth at the result limit.
        raise OperationResourceAdmissionError(
            location=("dfa", "word_length"),
            code="regular_language.count_intermediate_bound",
            message="DFA matrix-power intermediates exceed the canonical digit bound",
        )
    # Charge bit-work from the path-sensitive powered matrix, not max-row**n.
    # A 32-way transient that fires once per cycle stays far below 32**length.
    coefficient_bound = max(1, max_intermediate)
    matrix_bit_work = (
        state_count**3
        * max(1, word_length.bit_length())
        * max(1, coefficient_bound.bit_length())
    )
    if matrix_bit_work > MAX_COUNT_MATRIX_BIT_WORK:
        raise OperationResourceAdmissionError(
            location=("dfa", "word_length"),
            code="regular_language.count_work_bound",
            message="DFA matrix powering exceeds the exact work bound",
        )
    # Neither saturation bound was reached, so the admitted selected entry
    # is already exact. Reuse it instead of powering the matrix a second time.
    return selected_count


def _accepted_path_matrix(
    dfa: DFA,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]] | None:
    transitions = _transition_map(dfa)
    reachable = {dfa.initial_state}
    frontier = [dfa.initial_state]
    while frontier:
        source = frontier.pop()
        for symbol in range(dfa.alphabet_size):
            target = transitions[(source, symbol)]
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)

    predecessors: list[set[int]] = [set() for _ in range(dfa.state_count)]
    for (source, _symbol), target in transitions.items():
        predecessors[target].add(source)
    coreachable = set(dfa.accepting_states)
    frontier = list(dfa.accepting_states)
    while frontier:
        target = frontier.pop()
        for source in predecessors[target]:
            if source not in coreachable:
                coreachable.add(source)
                frontier.append(source)
    if dfa.initial_state not in coreachable:
        return None
    useful = reachable & coreachable
    states = (dfa.initial_state, *sorted(useful - {dfa.initial_state}))
    index = {state: position for position, state in enumerate(states)}
    matrix = [[0] * len(states) for _ in states]
    for (source, _symbol), target in _transition_map(dfa).items():
        if source in useful and target in useful:
            matrix[index[source]][index[target]] += 1
    accepting_states = tuple(
        index[state] for state in states if state in dfa.accepting_states
    )
    return tuple(tuple(row) for row in matrix), accepting_states


def _matrix_max_entry(matrix: tuple[tuple[int, ...], ...]) -> int:
    return max(max(row) for row in matrix)


def _powered_count_admission(
    matrix: tuple[tuple[int, ...], ...],
    accepting_states: tuple[int, ...],
    exponent: int,
    cap: int,
) -> tuple[int, int]:
    """Return capped ``(max materialized entry, selected count)`` of ``matrix ** exponent``.

    The maximum covers every matrix binary exponentiation materializes: the
    successive squares and the running product. Hitting ``cap`` means the true
    value is at least ``cap``.
    """

    size = len(matrix)
    running = tuple(tuple(int(i == j) for j in range(size)) for i in range(size))
    power = matrix
    max_entry = _matrix_max_entry(power)
    while exponent:
        if exponent & 1:
            running = _capped_matrix_product(running, power, cap)
            max_entry = max(max_entry, _matrix_max_entry(running))
        exponent >>= 1
        if exponent:
            power = _capped_matrix_product(power, power, cap)
            max_entry = max(max_entry, _matrix_max_entry(power))
    selected = min(sum(running[0][state] for state in accepting_states), cap)
    return max_entry, selected


def _capped_matrix_product(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    cap: int,
) -> tuple[tuple[int, ...], ...]:
    from flint import fmpz_mat

    size = len(left)
    product = fmpz_mat(left) * fmpz_mat(right)
    return tuple(
        tuple(min(int(product[row, column]), cap) for column in range(size))
        for row in range(size)
    )


def dfa_complement(dfa: DFA) -> DFA:
    """Compute the complement DFA by flipping the accepting states."""

    accepting = set(dfa.accepting_states)
    return DFA(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        alphabet_id=dfa.alphabet_id,
        alphabet=dfa.alphabet,
        transitions=dfa.transitions,
        initial_state=dfa.initial_state,
        accepting_states=tuple(sorted(set(range(dfa.state_count)) - accepting)),
    )


def dfa_subsequential_preimage(dfa: DFA, transducer: SubsequentialTransducer) -> DFA:
    """Return the input language mapped by the subsequential machine into L(dfa).

    Undefined transducer transitions and nonfinal terminal states reject. The
    transition and final output words are consumed by the DFA in order.
    """
    if type(dfa) is not DFA or type(transducer) is not SubsequentialTransducer:
        raise OperationDomainValidationError(
            location=("dfa", "transducer"),
            code="regular_language.preimage.noncanonical_input",
            message="preimage requires canonical DFA and subsequential transducer values",
        )
    dfa = _admit_cross_domain_dfa(dfa, "preimage")
    transducer = _admit_cross_domain_transducer(transducer, "preimage")
    if (
        dfa.alphabet is None
        or transducer.output_alphabet is None
        or transducer.input_alphabet is None
        or dfa.alphabet != transducer.output_alphabet
        or dfa.alphabet_id != transducer.output_alphabet_id
        or dfa.alphabet_size != len(dfa.alphabet.symbols)
    ):
        raise OperationDomainValidationError(
            location=("dfa", "transducer"),
            code="regular_language.preimage.alphabet_mismatch",
            message=(
                "preimage requires matching explicit output alphabet parents and "
                "an explicit input parent"
            ),
        )
    work_bound, output_transition_bound = _admit_subsequential_preimage(dfa, transducer)
    return _build_subsequential_preimage(
        dfa, transducer, work_bound, output_transition_bound
    )


def _admit_subsequential_preimage(
    dfa: DFA, transducer: SubsequentialTransducer
) -> tuple[int, int]:
    request_checkpoint("before subsequential preimage admission")
    input_size = transducer.input_alphabet_size
    product_state_bound = transducer.state_count * dfa.state_count
    # Any omitted transducer transition may require one additional rejecting
    # sink in the total DFA result. Reserve it before product exploration.
    has_undefined_transitions = len(transducer.transitions) < (
        transducer.state_count * input_size
    )
    product_bound = product_state_bound + int(has_undefined_transitions)
    product_transition_bound = product_bound * input_size
    max_output = max(
        (
            len(item.output)
            for item in (*transducer.transitions, *transducer.final_outputs)
        ),
        default=0,
    )
    work_bound = product_bound * (input_size + max_output * (input_size + 1))
    output_transition_bound = MAX_DFA_STATES * input_size
    output_bytes_bound = 4096 + output_transition_bound * 128
    if (
        product_bound > MAX_DFA_PREIMAGE_PRODUCT_STATES
        or product_transition_bound > MAX_DFA_PREIMAGE_PRODUCT_TRANSITIONS
        or work_bound > MAX_DFA_PREIMAGE_WORK
        or output_transition_bound > MAX_DFA_TRANSITIONS
        or output_bytes_bound > MAX_DFA_PREIMAGE_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("transducer", "dfa"),
            code="regular_language.preimage_resource_bound",
            message="subsequential preimage exceeds its product, work, or output bound",
        )
    # Admission uses the complete product bound, not the reachable subset, so
    # every accepted request fits the result carrier before expansion starts.
    request_checkpoint("after subsequential preimage admission")
    return work_bound, output_transition_bound


def _admit_cross_domain_dfa(dfa: DFA, operation: str) -> DFA:
    """Recheck a caller-supplied carrier at this public native boundary."""
    try:
        return DFA.model_validate(dfa.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("dfa",),
            code=f"regular_language.{operation}.invalid_dfa_carrier",
            message=f"{operation} requires a structurally valid canonical DFA",
        ) from exc


def _admit_cross_domain_nfa(nfa: NFA, operation: str) -> NFA:
    """Recheck a caller-supplied NFA at this public native boundary."""
    try:
        return NFA.model_validate(nfa.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("nfa",),
            code=f"regular_language.{operation}.invalid_nfa_carrier",
            message=f"{operation} requires a structurally valid canonical NFA",
        ) from exc


def _admit_cross_domain_transducer(
    transducer: SubsequentialTransducer, operation: str
) -> SubsequentialTransducer:
    """Recheck a caller-supplied carrier without replaying its computation."""
    try:
        return SubsequentialTransducer.model_validate(
            transducer.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("transducer",),
            code=f"regular_language.{operation}.invalid_transducer_carrier",
            message=(
                f"{operation} requires a structurally valid canonical "
                "subsequential transducer"
            ),
        ) from exc


def _build_subsequential_preimage(
    dfa: DFA,
    transducer: SubsequentialTransducer,
    work_bound: int,
    output_transition_bound: int,
) -> DFA:
    input_size = transducer.input_alphabet_size
    ledger = OperationWorkLedger(work_bound)
    transducer_edges = {
        (item.source, item.input_symbol): item for item in transducer.transitions
    }
    final_outputs = {item.state: item.output for item in transducer.final_outputs}
    dfa_edges = _transition_map(dfa)
    initial = (transducer.initial_state, dfa.initial_state)
    pairs = [initial]
    state_ids = {initial: 0}
    row_coordinates: list[tuple[int, int, int]] = []
    accepting: list[int] = []
    has_sink = False

    cursor = 0
    while cursor < len(pairs):
        request_checkpoint("during subsequential preimage product exploration")
        transducer_state, dfa_state = pairs[cursor]
        if transducer_state in final_outputs:
            terminal = _consume_dfa_output(
                dfa_edges,
                dfa_state,
                final_outputs[transducer_state],
                ledger,
            )
            if terminal in dfa.accepting_states:
                accepting.append(cursor)
        for symbol in range(input_size):
            ledger.charge()
            edge = transducer_edges.get((transducer_state, symbol))
            if edge is None:
                has_sink = True
                target_id = -1
            else:
                target = (
                    edge.target,
                    _consume_dfa_output(dfa_edges, dfa_state, edge.output, ledger),
                )
                target_id = state_ids.get(target, -1)
                if target_id < 0:
                    target_id = len(pairs)
                    state_ids[target] = target_id
                    pairs.append(target)
                    if len(pairs) + int(has_sink) > MAX_DFA_STATES:
                        raise OperationResourceAdmissionError(
                            location=("transducer", "dfa"),
                            code="regular_language.preimage_state_bound",
                            message="reachable subsequential preimage exceeds DFA state bound",
                        )
            row_coordinates.append((cursor, symbol, target_id))
        cursor += 1
    # A rejecting sink is needed whenever an undefined edge was seen. It loops
    # over all input symbols; its state was reserved above by the output bound.
    if len(row_coordinates) > output_transition_bound:
        raise OperationResourceAdmissionError(
            location=("transducer", "dfa"),
            code="regular_language.preimage_transition_bound",
            message="subsequential preimage exceeds DFA transition bound",
        )
    if has_sink:
        sink = len(pairs)
        row_coordinates = [
            (source, symbol, sink if target == -1 else target)
            for source, symbol, target in row_coordinates
        ]
        for symbol in range(input_size):
            row_coordinates.append((sink, symbol, sink))
    if len(pairs) + int(has_sink) > MAX_DFA_STATES:
        raise OperationResourceAdmissionError(
            location=("transducer", "dfa"),
            code="regular_language.preimage_state_bound",
            message="reachable subsequential preimage exceeds DFA state bound",
        )
    request_checkpoint("before subsequential preimage result construction")
    return DFA(
        state_count=len(pairs) + int(has_sink),
        alphabet_size=input_size,
        alphabet_id=transducer.input_alphabet_id,
        alphabet=transducer.input_alphabet,
        transitions=tuple(
            DFATransition(source=source, symbol=symbol, target=target)
            for source, symbol, target in row_coordinates
        ),
        initial_state=0,
        accepting_states=tuple(accepting),
    )


def _consume_dfa_output(
    dfa_edges: dict[tuple[int, int], int],
    state: int,
    output: tuple[int, ...],
    ledger: OperationWorkLedger,
) -> int:
    for output_symbol in output:
        ledger.charge()
        state = dfa_edges[(state, output_symbol)]
    return state


def dfa_subsequential_image(dfa: DFA, transducer: SubsequentialTransducer) -> NFA:
    """Return an epsilon-NFA for the transducer image of ``L(dfa)``.

    The construction expands reachable source-DFA/transducer product edges
    into an epsilon-NFA over output symbols, includes final-output paths only
    at accepting source states. Undefined transducer transitions contribute no
    path.
    """
    if type(dfa) is not DFA or type(transducer) is not SubsequentialTransducer:
        raise OperationDomainValidationError(
            location=("dfa", "transducer"),
            code="regular_language.image.noncanonical_input",
            message="image requires canonical DFA and subsequential transducer values",
        )
    dfa = _admit_cross_domain_dfa(dfa, "image")
    transducer = _admit_cross_domain_transducer(transducer, "image")
    if (
        dfa.alphabet is None
        or transducer.input_alphabet is None
        or transducer.output_alphabet is None
        or dfa.alphabet != transducer.input_alphabet
        or dfa.alphabet_id != transducer.input_alphabet_id
        or dfa.alphabet_size != len(dfa.alphabet.symbols)
    ):
        raise OperationDomainValidationError(
            location=("dfa", "transducer"),
            code="regular_language.image.alphabet_mismatch",
            message=(
                "image requires matching explicit input alphabets and an explicit "
                "transducer output alphabet"
            ),
        )
    work_bound, nfa_state_bound, nfa_transition_bound = _admit_subsequential_image(
        dfa, transducer
    )
    return _build_subsequential_image(
        dfa,
        transducer,
        work_bound,
        nfa_state_bound,
        nfa_transition_bound,
    )


def nfa_subsequential_image(nfa: NFA, transducer: SubsequentialTransducer) -> NFA:
    """Return an epsilon-NFA for the image of an epsilon-NFA language.

    The product retains source epsilon edges without advancing the transducer;
    labeled source edges advance both machines and emit the transducer output.
    Final outputs are included only at accepting source states.
    """
    if type(nfa) is not NFA or type(transducer) is not SubsequentialTransducer:
        raise OperationDomainValidationError(
            location=("nfa", "transducer"),
            code="regular_language.nfa_image.noncanonical_input",
            message="NFA image requires canonical NFA and subsequential transducer values",
        )
    nfa = _admit_cross_domain_nfa(nfa, "nfa_image")
    transducer = _admit_cross_domain_transducer(transducer, "nfa_image")
    if (
        nfa.alphabet is None
        or transducer.input_alphabet is None
        or transducer.output_alphabet is None
        or nfa.alphabet != transducer.input_alphabet
        or nfa.alphabet_id != transducer.input_alphabet_id
        or nfa.alphabet_size != len(nfa.alphabet.symbols)
    ):
        raise OperationDomainValidationError(
            location=("nfa", "transducer"),
            code="regular_language.nfa_image.alphabet_mismatch",
            message=(
                "NFA image requires matching explicit input alphabets and an "
                "explicit transducer output alphabet"
            ),
        )
    work_bound, state_bound, transition_bound = _admit_nfa_subsequential_image(
        nfa, transducer
    )
    return _build_nfa_subsequential_image(
        nfa, transducer, work_bound, state_bound, transition_bound
    )


def _admit_nfa_subsequential_image(
    nfa: NFA, transducer: SubsequentialTransducer
) -> tuple[int, int, int]:
    """Bound the full product and expanded output NFA before construction."""
    request_checkpoint("before NFA subsequential image admission")
    pair_bound = nfa.state_count * transducer.state_count
    epsilon_edges = sum(edge.symbol is None for edge in nfa.transitions)
    symbol_edge_counts = [0] * nfa.alphabet_size
    for edge in nfa.transitions:
        if edge.symbol is not None:
            symbol_edge_counts[edge.symbol] += 1
    transducer_edges_by_symbol: dict[int, list[int]] = {}
    for edge in transducer.transitions:
        transducer_edges_by_symbol.setdefault(edge.input_symbol, []).append(
            len(edge.output)
        )
    product_edge_bound = epsilon_edges * transducer.state_count + sum(
        source_count * len(transducer_edges_by_symbol.get(symbol, ()))
        for symbol, source_count in enumerate(symbol_edge_counts)
    )
    output_transition_bound = epsilon_edges * transducer.state_count + sum(
        source_count
        * sum(
            max(1, output_length)
            for output_length in transducer_edges_by_symbol.get(symbol, ())
        )
        for symbol, source_count in enumerate(symbol_edge_counts)
    )
    output_intermediates = sum(
        source_count
        * sum(
            max(0, output_length - 1)
            for output_length in transducer_edges_by_symbol.get(symbol, ())
        )
        for symbol, source_count in enumerate(symbol_edge_counts)
    )
    terminal_pair_bound = len(nfa.accepting_states) * len(transducer.final_outputs)
    final_output_transitions = len(nfa.accepting_states) * sum(
        max(1, len(final.output)) for final in transducer.final_outputs
    )
    final_intermediates = len(nfa.accepting_states) * sum(
        max(0, len(final.output) - 1) for final in transducer.final_outputs
    )
    state_bound = (
        pair_bound
        + output_intermediates
        + final_intermediates
        + int(terminal_pair_bound > 0)
    )
    transition_bound = output_transition_bound + final_output_transitions
    work_bound = (
        pair_bound
        + len(nfa.transitions) * transducer.state_count
        + output_transition_bound
        + final_output_transitions
        + state_bound
        + nfa.state_count
        + len(nfa.accepting_states)
    )
    intermediate_bytes_bound = (
        pair_bound * 128
        + product_edge_bound * 128
        + terminal_pair_bound * 96
        + state_bound * 256
        + transition_bound * 128
        + nfa.state_count
    )
    output_bytes_bound = 4096 + state_bound * 24 + transition_bound * 72
    if (
        state_bound > MAX_NFA_STATES
        or transition_bound > MAX_NFA_TRANSITIONS
        or intermediate_bytes_bound > MAX_SUBSEQUENTIAL_IMAGE_INTERMEDIATE_BYTES
        or output_bytes_bound > MAX_SUBSEQUENTIAL_IMAGE_OUTPUT_BYTES
        or work_bound > MAX_SUBSEQUENTIAL_IMAGE_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("nfa", "transducer"),
            code="regular_language.nfa_image_resource_bound",
            message=(
                "subsequential NFA image exceeds its product or output-NFA "
                "work, intermediate-allocation, or output bound"
            ),
        )
    request_checkpoint("after NFA subsequential image admission")
    return work_bound, state_bound, transition_bound


def _build_nfa_subsequential_image(
    nfa: NFA,
    transducer: SubsequentialTransducer,
    work_bound: int,
    state_bound: int,
    transition_bound: int,
) -> NFA:
    ledger = OperationWorkLedger(work_bound)
    ledger.charge(nfa.state_count + len(nfa.accepting_states))
    accepting_states = bytearray(nfa.state_count)
    for state in nfa.accepting_states:
        accepting_states[state] = 1
    outgoing: list[list[tuple[int | None, int]]] = [[] for _ in range(nfa.state_count)]
    for edge in nfa.transitions:
        outgoing[edge.source].append((edge.symbol, edge.target))
    transducer_edges = {
        (edge.source, edge.input_symbol): edge for edge in transducer.transitions
    }
    final_outputs = {final.state: final.output for final in transducer.final_outputs}
    initial_pair = (nfa.initial_state, transducer.initial_state)
    pairs = [initial_pair]
    pair_ids = {initial_pair: 0}
    product_edges: list[tuple[int, tuple[int, ...], int]] = []
    terminal_pairs: list[tuple[int, tuple[int, ...]]] = []
    cursor = 0
    while cursor < len(pairs):
        request_checkpoint("during subsequential NFA image product exploration")
        source_state, transducer_state = pairs[cursor]
        final_output = final_outputs.get(transducer_state)
        if accepting_states[source_state] and final_output is not None:
            terminal_pairs.append((cursor, final_output))
        for input_symbol, source_target in outgoing[source_state]:
            ledger.charge()
            if input_symbol is None:
                edge_output: tuple[int, ...] = ()
                target_transducer_state = transducer_state
            else:
                transition = transducer_edges.get((transducer_state, input_symbol))
                if transition is None:
                    continue
                edge_output = transition.output
                target_transducer_state = transition.target
            target_pair = (source_target, target_transducer_state)
            target_id = pair_ids.get(target_pair)
            if target_id is None:
                target_id = len(pairs)
                pair_ids[target_pair] = target_id
                pairs.append(target_pair)
            product_edges.append((cursor, edge_output, target_id))
            ledger.charge(len(edge_output))
        cursor += 1
    transitions, state_count, accepting_sink = _expand_image_nfa(
        pairs,
        product_edges,
        terminal_pairs,
        state_bound,
        transition_bound,
        ledger,
    )
    request_checkpoint("before subsequential NFA image result construction")
    return NFA(
        state_count=state_count,
        alphabet_size=transducer.output_alphabet_size,
        alphabet_id=transducer.output_alphabet_id,
        alphabet=transducer.output_alphabet,
        transitions=transitions,
        initial_state=0,
        accepting_states=() if accepting_sink is None else (accepting_sink,),
    )


def _admit_subsequential_image(
    dfa: DFA, transducer: SubsequentialTransducer
) -> tuple[int, int, int]:
    request_checkpoint("before subsequential image admission")
    pair_bound = dfa.state_count * transducer.state_count
    transition_output_edges = dfa.state_count * sum(
        max(1, len(edge.output)) for edge in transducer.transitions
    )
    final_output_edges = len(dfa.accepting_states) * sum(
        max(1, len(final.output)) for final in transducer.final_outputs
    )
    transition_intermediates = dfa.state_count * sum(
        max(0, len(edge.output) - 1) for edge in transducer.transitions
    )
    final_intermediates = len(dfa.accepting_states) * sum(
        max(0, len(final.output) - 1) for final in transducer.final_outputs
    )
    nfa_state_bound = pair_bound + transition_intermediates + final_intermediates + 1
    nfa_transition_bound = transition_output_edges + final_output_edges
    product_transition_bound = dfa.state_count * len(transducer.transitions)
    terminal_pair_bound = len(dfa.accepting_states) * len(transducer.final_outputs)
    intermediate_bytes_bound = (
        pair_bound * 128
        + product_transition_bound * 128
        + terminal_pair_bound * 96
        + nfa_state_bound * 256
        + nfa_transition_bound * 128
    )
    output_bytes_bound = 4096 + nfa_state_bound * 24 + nfa_transition_bound * 72
    work_bound = (
        pair_bound * transducer.input_alphabet_size
        + transition_output_edges
        + final_output_edges
        + nfa_state_bound
        + nfa_transition_bound
    )
    if (
        nfa_state_bound > MAX_NFA_STATES
        or nfa_transition_bound > MAX_NFA_TRANSITIONS
        or intermediate_bytes_bound > MAX_SUBSEQUENTIAL_IMAGE_INTERMEDIATE_BYTES
        or output_bytes_bound > MAX_SUBSEQUENTIAL_IMAGE_OUTPUT_BYTES
        or work_bound > MAX_SUBSEQUENTIAL_IMAGE_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("dfa", "transducer"),
            code="regular_language.image_resource_bound",
            message=(
                "subsequential image exceeds its product or epsilon-NFA "
                "work, intermediate-allocation, or output bound"
            ),
        )
    request_checkpoint("after subsequential image admission")
    return work_bound, nfa_state_bound, nfa_transition_bound


def _build_subsequential_image(
    dfa: DFA,
    transducer: SubsequentialTransducer,
    work_bound: int,
    nfa_state_bound: int,
    nfa_transition_bound: int,
) -> NFA:
    ledger = OperationWorkLedger(work_bound)
    pairs, product_edges, terminal_pairs = _reachable_image_product(
        dfa, transducer, ledger
    )
    transitions, state_count, accepting_sink = _expand_image_nfa(
        pairs,
        product_edges,
        terminal_pairs,
        nfa_state_bound,
        nfa_transition_bound,
        ledger,
    )
    request_checkpoint("before subsequential image result construction")
    return NFA(
        state_count=state_count,
        alphabet_size=transducer.output_alphabet_size,
        alphabet_id=transducer.output_alphabet_id,
        alphabet=transducer.output_alphabet,
        transitions=transitions,
        initial_state=0,
        accepting_states=() if accepting_sink is None else (accepting_sink,),
    )


def _reachable_image_product(
    dfa: DFA,
    transducer: SubsequentialTransducer,
    ledger: OperationWorkLedger,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, tuple[int, ...], int]],
    list[tuple[int, tuple[int, ...]]],
]:
    input_edges = _transition_map(dfa)
    transducer_edges = {
        (edge.source, edge.input_symbol): edge for edge in transducer.transitions
    }
    finals = {item.state: item.output for item in transducer.final_outputs}
    initial_pair = (dfa.initial_state, transducer.initial_state)
    pairs = [initial_pair]
    pair_ids = {initial_pair: 0}
    product_edges: list[tuple[int, tuple[int, ...], int]] = []
    terminal_pairs: list[tuple[int, tuple[int, ...]]] = []
    cursor = 0
    while cursor < len(pairs):
        request_checkpoint("during subsequential image product exploration")
        dfa_state, transducer_state = pairs[cursor]
        final_output = finals.get(transducer_state)
        if dfa_state in dfa.accepting_states and final_output is not None:
            terminal_pairs.append((cursor, final_output))
        for input_symbol in range(dfa.alphabet_size):
            ledger.charge()
            edge = transducer_edges.get((transducer_state, input_symbol))
            if edge is None:
                continue
            target_pair = (
                input_edges[(dfa_state, input_symbol)],
                edge.target,
            )
            target_id = pair_ids.get(target_pair)
            if target_id is None:
                target_id = len(pairs)
                pair_ids[target_pair] = target_id
                pairs.append(target_pair)
            product_edges.append((cursor, edge.output, target_id))
            ledger.charge(len(edge.output))
        cursor += 1

    return pairs, product_edges, terminal_pairs


def _expand_image_nfa(
    pairs: list[tuple[int, int]],
    product_edges: list[tuple[int, tuple[int, ...], int]],
    terminal_pairs: list[tuple[int, tuple[int, ...]]],
    nfa_state_bound: int,
    nfa_transition_bound: int,
    ledger: OperationWorkLedger,
) -> tuple[tuple[NFATransition, ...], int, int | None]:
    state_count = len(pairs)
    transitions: list[NFATransition] = []

    def add_nfa_state() -> int:
        nonlocal state_count
        if state_count >= nfa_state_bound:
            raise OperationResourceAdmissionError(
                location=("dfa", "transducer"),
                code="regular_language.image_state_bound",
                message="subsequential image exceeds NFA state bound",
            )
        ledger.charge()
        state = state_count
        state_count += 1
        return state

    def add_transition(source: int, symbol: int | None, target: int) -> None:
        if len(transitions) >= nfa_transition_bound:
            raise OperationResourceAdmissionError(
                location=("dfa", "transducer"),
                code="regular_language.image_transition_bound",
                message="subsequential image exceeds NFA transition bound",
            )
        ledger.charge()
        transitions.append(
            NFATransition(
                transition_id=len(transitions),
                source=source,
                symbol=symbol,
                target=target,
            )
        )

    # Product state IDs are stable as BFS discovers them.
    accepting_sink = add_nfa_state() if terminal_pairs else None

    def add_word_path(source: int, word: tuple[int, ...], target: int) -> None:
        if not word:
            add_transition(source, None, target)
            return
        current = source
        for offset, symbol in enumerate(word):
            next_state = target if offset == len(word) - 1 else add_nfa_state()
            add_transition(current, symbol, next_state)
            current = next_state

    for source, output, target in product_edges:
        request_checkpoint("during subsequential image path expansion")
        add_word_path(source, output, target)
    for source, output in terminal_pairs:
        request_checkpoint("during subsequential image final-output expansion")
        assert accepting_sink is not None
        add_word_path(source, output, accepting_sink)
    return tuple(transitions), state_count, accepting_sink


def _require_equivalence_dfa(value: object, location: str) -> int:
    """Guard the native entry point against unvalidated or partial DFAs."""

    if type(value) is not DFA:
        raise OperationDomainValidationError(
            location=(location,),
            code=f"regular_language.equivalence.{location}_not_dfa",
            message=f"{location} must be a canonical DFA value",
        )
    state_count = getattr(value, "state_count", None)
    alphabet_size = getattr(value, "alphabet_size", None)
    initial_state = getattr(value, "initial_state", None)
    accepting_states = getattr(value, "accepting_states", None)
    transitions = getattr(value, "transitions", None)
    if (
        type(state_count) is not int
        or not 1 <= state_count <= MAX_DFA_STATES
        or type(alphabet_size) is not int
        or not 0 <= alphabet_size <= MAX_DFA_ALPHABET
        or type(initial_state) is not int
        or not 0 <= initial_state < state_count
        or type(accepting_states) is not tuple
        or len(accepting_states) > MAX_DFA_STATES
        or any(
            type(state) is not int or not 0 <= state < state_count
            for state in accepting_states
        )
        or len(set(accepting_states)) != len(accepting_states)
        or type(transitions) is not tuple
        or len(transitions) > MAX_DFA_TRANSITIONS
    ):
        raise OperationDomainValidationError(
            location=(location,),
            code="regular_language.equivalence.invalid_dfa",
            message=f"{location} is not a structurally valid DFA",
        )
    keys: set[tuple[int, int]] = set()
    for transition in transitions:
        source = getattr(transition, "source", None)
        symbol = getattr(transition, "symbol", None)
        target = getattr(transition, "target", None)
        if (
            type(transition) is not DFATransition
            or type(source) is not int
            or type(symbol) is not int
            or type(target) is not int
            or not 0 <= source < state_count
            or not 0 <= target < state_count
            or not 0 <= symbol < alphabet_size
        ):
            raise OperationDomainValidationError(
                location=(location, "transitions"),
                code="regular_language.equivalence.invalid_dfa",
                message=f"{location} contains an invalid transition",
            )
        key = (source, symbol)
        if key in keys:
            raise OperationDomainValidationError(
                location=(location, "transitions"),
                code="regular_language.equivalence.partial_dfa",
                message=f"{location} must have one transition for every state-symbol pair",
            )
        keys.add(key)
    expected = state_count * alphabet_size
    if len(keys) != expected:
        raise OperationDomainValidationError(
            location=(location, "transitions"),
            code="regular_language.equivalence.partial_dfa",
            message=f"{location} must have one transition for every state-symbol pair",
        )
    return len(transitions) + len(accepting_states) + 4


def _admit_dfa_equivalence(left: DFA, right: DFA) -> tuple[int, int]:
    """Admit every BFS materialization before the product search begins."""

    source_work = _require_equivalence_dfa(left, "left")
    source_work += _require_equivalence_dfa(right, "right")
    if (
        left.alphabet_size != right.alphabet_size
        or left.alphabet_id != right.alphabet_id
        or left.alphabet != right.alphabet
    ):
        raise OperationDomainValidationError(
            location=("right", "alphabet_size"),
            code="regular_language.equivalence.alphabet_mismatch",
            message="DFA equivalence requires the same ordered alphabet",
        )
    product_states = left.state_count * right.state_count
    product_transitions = product_states * left.alphabet_size
    witness_length = product_states - 1
    predecessor_allocation = product_states * 4
    trace_rows = 2 * (witness_length + 1)
    reconstruction_work = witness_length + trace_rows
    work = source_work + product_transitions + product_states + reconstruction_work
    output_allocation = (
        2
        * (
            left.state_count
            + right.state_count
            + left.alphabet_size
            + len(left.transitions)
            + len(right.transitions)
            + len(left.accepting_states)
            + len(right.accepting_states)
        )
        + witness_length
        + trace_rows
        + 8
    )
    bounds = (
        ("product_states", product_states, MAX_DFA_EQUIVALENCE_PRODUCT_STATES),
        (
            "product_transitions",
            product_transitions,
            MAX_DFA_EQUIVALENCE_PRODUCT_TRANSITIONS,
        ),
        ("witness_length", witness_length, MAX_DFA_EQUIVALENCE_WITNESS_LENGTH),
        ("trace_rows", trace_rows, MAX_DFA_EQUIVALENCE_TRACE_ROWS),
        (
            "intermediate_cells",
            predecessor_allocation,
            MAX_DFA_EQUIVALENCE_INTERMEDIATE_ALLOCATION,
        ),
        ("work", work, MAX_DFA_EQUIVALENCE_WORK),
        ("output_allocation", output_allocation, MAX_DFA_EQUIVALENCE_OUTPUT_ALLOCATION),
    )
    for resource, observed, admitted in bounds:
        if observed > admitted:
            raise OperationResourceAdmissionError(
                location=("left", "right"),
                code=f"regular_language.equivalence.{resource}_bound",
                message=(
                    f"DFA equivalence {resource} exceeds the admitted "
                    f"bound of {admitted}"
                ),
            )
    return source_work, work


def dfa_equivalence(
    left: DFA, right: DFA
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None, tuple[int, ...] | None]:
    """Return the shortest lexicographically least distinguishing word and traces."""

    request_checkpoint("before DFA equivalence admission")
    source_work, admitted_work = _admit_dfa_equivalence(left, right)
    request_checkpoint("after DFA equivalence admission")
    ledger = OperationWorkLedger(admitted_work)
    ledger.charge(source_work)
    left_transitions = _transition_map(left)
    right_transitions = _transition_map(right)
    initial = (left.initial_state, right.initial_state)
    queue = deque([initial])
    predecessor: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {
        initial: None
    }
    while queue:
        ledger.charge()
        request_checkpoint("during DFA equivalence product search")
        pair = queue.popleft()
        left_accepts = pair[0] in left.accepting_states
        right_accepts = pair[1] in right.accepting_states
        if left_accepts != right_accepts:
            pairs = [pair]
            symbols: list[int] = []
            link = predecessor[pairs[-1]]
            while link is not None:
                ledger.charge()
                request_checkpoint("during DFA equivalence witness reconstruction")
                parent, symbol = link
                symbols.append(symbol)
                pairs.append(parent)
                link = predecessor[parent]
            pairs.reverse()
            symbols.reverse()
            request_checkpoint("before DFA equivalence result construction")
            return (
                tuple(symbols),
                tuple(state[0] for state in pairs),
                tuple(state[1] for state in pairs),
            )
        for symbol in range(left.alphabet_size):
            ledger.charge()
            target = (
                left_transitions[(pair[0], symbol)],
                right_transitions[(pair[1], symbol)],
            )
            if target not in predecessor:
                predecessor[target] = (pair, symbol)
                queue.append(target)
    request_checkpoint("before DFA equivalence result construction")
    return None, None, None


def dfa_transition_carrier(dfa: DFA) -> FiniteLabeledAutomaton:
    """Project a DFA onto a labeled carrier with a stable transition axis."""

    ordered = sorted(dfa.transitions, key=lambda item: (item.source, item.symbol))
    return FiniteLabeledAutomaton(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        transitions=tuple(
            AutomatonTransition(
                transition_id=transition_id,
                source=transition.source,
                symbol=transition.symbol,
                target=transition.target,
            )
            for transition_id, transition in enumerate(ordered)
        ),
    )


def _transition_parikh_profile_data(
    plan: TransitionParikhAdmissionPlan,
    source_state: int,
    target_state: int,
    path_length: int,
) -> tuple[tuple[tuple[tuple[int, ...], int], ...], int]:
    """Compute canonical profile entries inside one admitted envelope."""

    if plan.expected_path_count == 0:
        return (), 0
    transition_count = plan.transition_count
    zero_vector = tuple(0 for _ in range(transition_count))
    layer: dict[tuple[int, tuple[int, ...]], int] = {(source_state, zero_vector): 1}
    for _ in range(path_length):
        if not layer:
            break
        next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
        for (state, vector), multiplicity in layer.items():
            for transition in plan.outgoing[state]:
                transition_id = transition.transition_id
                updated = (
                    *vector[:transition_id],
                    vector[transition_id] + 1,
                    *vector[transition_id + 1 :],
                )
                key = (transition.target, updated)
                next_layer[key] = next_layer.get(key, 0) + multiplicity
        layer = next_layer
    target_entries = sorted(
        (vector, multiplicity)
        for (state, vector), multiplicity in layer.items()
        if state == target_state
    )
    total_path_count = sum(multiplicity for _, multiplicity in target_entries)
    if total_path_count != plan.expected_path_count:
        raise RuntimeError(
            "transition-Parikh recurrence disagrees with independent path counting"
        )
    return tuple(target_entries), total_path_count


def transition_parikh_profile(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> TransitionParikhProfile:
    """Return the exact transition-use histogram for fixed-endpoint paths."""

    plan = admit_transition_profile(automaton, source_state, target_state, path_length)
    target_entries, total_path_count = _transition_parikh_profile_data(
        plan, source_state, target_state, path_length
    )
    return TransitionParikhProfile._from_kernel(
        automaton=automaton,
        source_state=source_state,
        target_state=target_state,
        path_length=path_length,
        entries=tuple(
            TransitionParikhCell(
                transition_counts=vector,
                multiplicity=multiplicity,
            )
            for vector, multiplicity in target_entries
        ),
        total_path_count=total_path_count,
    )


def verify_dfa_run(claim: RunResult) -> bool:
    """Verify DFA transitions, final state, and acceptance for a run claim."""

    transitions = _transition_map(claim.dfa)
    if len(claim.state_trace) != len(claim.word) + 1:
        return False
    state = claim.dfa.initial_state
    if claim.state_trace[0] != state:
        return False
    try:
        for symbol, observed in zip(claim.word, claim.state_trace[1:], strict=True):
            state = transitions[(state, symbol)]
            if observed != state:
                return False
    except (KeyError, ValueError, TypeError):
        return False
    return state == claim.final_state and claim.accepted == (
        state in claim.dfa.accepting_states
    )


def verify_accepted_word_count(claim: CountResult) -> bool:
    """Verify an exact accepted-word count for the retained DFA and length."""

    try:
        return count_accepted_words(claim.dfa, claim.word_length) == claim.count
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_transition_parikh_profile(claim: TransitionParikhProfile) -> bool:
    """Verify the complete transition-use histogram for a retained automaton."""

    try:
        return (
            transition_parikh_profile(
                claim.automaton,
                claim.source_state,
                claim.target_state,
                claim.path_length,
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
