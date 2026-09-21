"""Domain-owned finite-state transducer kernels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers._models import (
    ComposeResult,
    MinimizeResult,
    StatePairDistinguishability,
    SubseqRunResult,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


@dataclass(frozen=True, slots=True)
class _EquivalentStatePair:
    """Private outcome for a pair whose final-output behavior agrees."""

    first_state: int
    second_state: int


@dataclass(frozen=True, slots=True)
class _SeparatedStatePair:
    """Private outcome for a pair with a valid separating input word."""

    first_state: int
    second_state: int
    witness_word: tuple[int, ...]


_StatePairOutcome = _EquivalentStatePair | _SeparatedStatePair


__all__ = [
    "coaccessible_states",
    "compose_subsequential",
    "identity_transducer",
    "invert_rational",
    "minimize_subsequential",
    "reachable_states",
    "replay_rational_path",
    "run_subsequential",
    "trim_subsequential",
    "verify_composition",
    "verify_minimization",
    "verify_subsequential_run",
]


MAX_MINIMIZE_SAMPLE_WORDS = 20000


def _reject(code: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_state_transducer.{code}",
        message=message,
    )


def _transition_map(
    transducer: SubsequentialTransducer,
) -> dict[tuple[int, int], tuple[int, tuple[int, ...]]]:
    return {
        (tr.source, tr.input_symbol): (tr.target, tr.output)
        for tr in transducer.transitions
    }


def _final_output_map(
    transducer: SubsequentialTransducer,
) -> dict[int, tuple[int, ...]]:
    return {fo.state: fo.output for fo in transducer.final_outputs}


def run_subsequential(
    transducer: SubsequentialTransducer,
    word: tuple[int, ...],
) -> tuple[
    Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"],
    tuple[int, ...],
    int,
    int | None,
    tuple[int, ...],
]:
    """Run a subsequential transducer on ``word``.

    Returns ``(status, output, final_state, undefined_position, partial_output)``.

    ``status`` is one of ``"OUTPUT"``, ``"UNDEFINED_TRANSITION"``, or
    ``"NONFINAL_DOMAIN_STATE"``.
    """
    if any(not 0 <= symbol < transducer.input_alphabet_size for symbol in word):
        _reject(
            "word_symbol_out_of_range",
            "word symbol is outside the input alphabet",
            "word",
        )
    if len(word) > MAX_FST_WORD_LENGTH:
        _reject("word_length_exceeded", "input word exceeds the length bound", "word")
    transition_bound = max(
        (len(transition.output) for transition in transducer.transitions),
        default=0,
    )
    final_bound = max(
        (len(final.output) for final in transducer.final_outputs), default=0
    )
    if len(word) * transition_bound + final_bound > MAX_FST_RESULT_WORD_LENGTH:
        _reject(
            "run_output_exceeds_bound",
            "subsequential output may exceed the result word bound",
            "word",
        )
    transitions = _transition_map(transducer)
    finals = _final_output_map(transducer)
    state = transducer.initial_state
    accumulated: list[int] = []
    for pos, symbol in enumerate(word):
        key = (state, symbol)
        if key not in transitions:
            return (
                "UNDEFINED_TRANSITION",
                (),
                state,
                pos,
                tuple(accumulated),
            )
        target, output = transitions[key]
        accumulated.extend(output)
        if len(accumulated) > MAX_FST_RESULT_WORD_LENGTH:
            raise RuntimeError("admitted subsequential output exceeded its bound")
        state = target
    if state not in finals:
        return (
            "NONFINAL_DOMAIN_STATE",
            (),
            state,
            None,
            tuple(accumulated),
        )
    final_word = finals[state]
    accumulated.extend(final_word)
    if len(accumulated) > MAX_FST_RESULT_WORD_LENGTH:
        raise RuntimeError("admitted subsequential output exceeded its bound")
    return ("OUTPUT", tuple(accumulated), state, None, ())


def identity_transducer(alphabet_size: int) -> SubsequentialTransducer:
    """Return the identity subsequential transducer on one alphabet."""

    transitions = tuple(
        SubseqTransition(
            source=0,
            input_symbol=sym,
            target=0,
            output=(sym,),
        )
        for sym in range(alphabet_size)
    )
    return SubsequentialTransducer(
        input_alphabet_size=alphabet_size,
        output_alphabet_size=alphabet_size,
        state_count=1,
        initial_state=0,
        transitions=transitions,
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )


def reachable_states(
    transducer: SubsequentialTransducer,
) -> set[int]:
    """Return states reachable from the initial state by defined transitions."""

    visited: set[int] = set()
    queue: deque[int] = deque([transducer.initial_state])
    visited.add(transducer.initial_state)
    adj: dict[int, list[tuple[int, int]]] = {}
    for tr in transducer.transitions:
        adj.setdefault(tr.source, []).append((tr.input_symbol, tr.target))
    while queue:
        current = queue.popleft()
        for _sym, target in adj.get(current, []):
            if target not in visited:
                visited.add(target)
                queue.append(target)
    return visited


def coaccessible_states(
    transducer: SubsequentialTransducer,
) -> set[int]:
    """Return states from which some final-output state is reachable."""

    finals = {fo.state for fo in transducer.final_outputs}
    reverse_adj: dict[int, list[int]] = {}
    for tr in transducer.transitions:
        reverse_adj.setdefault(tr.target, []).append(tr.source)
    visited: set[int] = set()
    queue: deque[int] = deque(finals)
    visited.update(finals)
    while queue:
        current = queue.popleft()
        for source in reverse_adj.get(current, []):
            if source not in visited:
                visited.add(source)
                queue.append(source)
    return visited


def _admit_trim(transducer: SubsequentialTransducer) -> None:
    if not isinstance(transducer, SubsequentialTransducer):
        _reject(
            "transducer_type",
            "transducer must be a SubsequentialTransducer value",
            "transducer",
        )


def trim_subsequential(
    transducer: SubsequentialTransducer,
) -> tuple[SubsequentialTransducer, dict[int, int]]:
    """Restrict a transducer to reachable and coaccessible states.

    Returns the trimmed transducer and an old-state -> new-state map.
    """
    _admit_trim(transducer)
    reachable = reachable_states(transducer)
    coaccessible = coaccessible_states(transducer)
    keep = reachable & coaccessible
    if not keep:
        return (
            SubsequentialTransducer(
                input_alphabet_size=transducer.input_alphabet_size,
                output_alphabet_size=transducer.output_alphabet_size,
                state_count=1,
                initial_state=0,
                transitions=(),
                final_outputs=(),
            ),
            {},
        )
    old_to_new = {old: new for new, old in enumerate(sorted(keep))}
    new_transitions = tuple(
        SubseqTransition(
            source=old_to_new[tr.source],
            input_symbol=tr.input_symbol,
            target=old_to_new[tr.target],
            output=tr.output,
        )
        for tr in transducer.transitions
        if tr.source in keep and tr.target in keep
    )
    new_finals = tuple(
        SubseqFinalOutput(
            state=old_to_new[fo.state],
            output=fo.output,
        )
        for fo in transducer.final_outputs
        if fo.state in keep
    )
    return (
        SubsequentialTransducer(
            input_alphabet_size=transducer.input_alphabet_size,
            output_alphabet_size=transducer.output_alphabet_size,
            state_count=len(keep),
            initial_state=old_to_new[transducer.initial_state],
            transitions=new_transitions,
            final_outputs=new_finals,
        ),
        old_to_new,
    )


def _run_u_on_word(
    u_map: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    u_state: int,
    word: tuple[int, ...],
) -> tuple[int, list[int]] | None:
    """Run U over a finite word starting from u_state.

    Returns ``(final_u_state, output)`` or ``None`` if U is undefined.
    """
    u_current = u_state
    all_output: list[int] = []
    for out_sym in word:
        u_key = (u_current, out_sym)
        if u_key not in u_map:
            return None
        u_next, u_output = u_map[u_key]
        all_output.extend(u_output)
        u_current = u_next
    return (u_current, all_output)


def compose_subsequential(
    first: SubsequentialTransducer,
    second: SubsequentialTransducer,
) -> SubsequentialTransducer:
    """Compose two subsequential transducers, computing U o T.

    The result computes ``second(first(word))`` whenever both are defined.
    Exploration admits at most 64 reachable pairs, each with at most 32 input
    symbols and 512 intermediate symbols. Unreachable Cartesian pairs need
    neither storage nor transition work.
    """
    _validate_composition_bounds(first, second)
    t_map = {
        (tr.source, tr.input_symbol): (tr.target, tr.output) for tr in first.transitions
    }
    t_finals = {fo.state: fo.output for fo in first.final_outputs}
    u_map = {
        (tr.source, tr.input_symbol): (tr.target, tr.output)
        for tr in second.transitions
    }
    u_finals = {fo.state: fo.output for fo in second.final_outputs}

    start_pair = (first.initial_state, second.initial_state)
    state_pairs: dict[tuple[int, int], int] = {start_pair: 0}
    queue: deque[tuple[int, int]] = deque([start_pair])
    new_transitions: list[SubseqTransition] = []

    while queue:
        t_state, u_state = queue.popleft()
        new_state = state_pairs[(t_state, u_state)]
        for sym in range(first.input_alphabet_size):
            key = (t_state, sym)
            if key not in t_map:
                continue
            t_next, t_output = t_map[key]
            result = _run_u_on_word(u_map, u_state, t_output)
            if result is None:
                continue
            u_next_final, all_output = result
            if len(all_output) > MAX_FST_WORD_LENGTH:
                raise RuntimeError("admitted composite transition exceeded its bound")
            pair_next = (t_next, u_next_final)
            if pair_next not in state_pairs:
                if len(state_pairs) == MAX_FST_STATES:
                    raise OperationResourceAdmissionError(
                        location=("first", "second"),
                        code="finite_state_transducer.composition_state_bound_exceeded",
                        message=f"reachable composite state count exceeds {MAX_FST_STATES}",
                    )
                state_pairs[pair_next] = len(state_pairs)
                queue.append(pair_next)
            new_state_next = state_pairs[pair_next]
            new_transitions.append(
                SubseqTransition(
                    source=new_state,
                    input_symbol=sym,
                    target=new_state_next,
                    output=tuple(all_output),
                )
            )

    new_finals = _composite_finals(state_pairs, t_finals, u_map, u_finals)

    return SubsequentialTransducer(
        input_alphabet_size=first.input_alphabet_size,
        output_alphabet_size=second.output_alphabet_size,
        state_count=len(state_pairs),
        initial_state=0,
        transitions=tuple(new_transitions),
        final_outputs=tuple(new_finals),
    )


def _composite_finals(
    state_pairs: dict[tuple[int, int], int],
    first_finals: dict[int, tuple[int, ...]],
    second_transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    second_finals: dict[int, tuple[int, ...]],
) -> list[SubseqFinalOutput]:
    result_finals: list[SubseqFinalOutput] = []
    for (first_state, second_state), composite_state in state_pairs.items():
        if first_state not in first_finals:
            continue
        result = _run_u_on_word(
            second_transitions, second_state, first_finals[first_state]
        )
        if result is None or result[0] not in second_finals:
            continue
        final_state, output = result
        output.extend(second_finals[final_state])
        if len(output) > MAX_FST_WORD_LENGTH:
            raise RuntimeError("admitted composite final output exceeded its bound")
        result_finals.append(
            SubseqFinalOutput(state=composite_state, output=tuple(output))
        )
    return result_finals


def _validate_composition_bounds(
    first: SubsequentialTransducer, second: SubsequentialTransducer
) -> None:
    if first.output_alphabet_size != second.input_alphabet_size:
        _reject(
            "composition_alphabet_mismatch",
            "first output alphabet must match second input alphabet",
            "first",
            "second",
        )
    second_output_bound = max(
        (len(transition.output) for transition in second.transitions), default=0
    )
    first_transition_bound = max(
        (len(transition.output) for transition in first.transitions), default=0
    )
    first_final_bound = max(
        (len(final.output) for final in first.final_outputs), default=0
    )
    second_final_bound = max(
        (len(final.output) for final in second.final_outputs), default=0
    )
    if first_transition_bound * second_output_bound > MAX_FST_WORD_LENGTH:
        _reject(
            "composition_transition_output_exceeds_bound",
            "composite transition output may exceed the word bound",
            "first",
            "second",
        )
    if (
        first_final_bound * second_output_bound + second_final_bound
        > MAX_FST_WORD_LENGTH
    ):
        _reject(
            "composition_final_output_exceeds_bound",
            "composite final output may exceed the word bound",
            "first",
            "second",
        )


def invert_rational(
    transducer: RationalTransducer,
) -> RationalTransducer:
    """Invert a rational transducer by swapping input/output labels and alphabets."""

    return RationalTransducer(
        input_alphabet_size=transducer.output_alphabet_size,
        output_alphabet_size=transducer.input_alphabet_size,
        state_count=transducer.state_count,
        initial_states=transducer.initial_states,
        accepting_states=transducer.accepting_states,
        edges=tuple(
            RationalEdge(
                source=e.source,
                target=e.target,
                input_label=e.output_label,
                output_label=e.input_label,
            )
            for e in transducer.edges
        ),
    )


def replay_rational_path(
    transducer: RationalTransducer,
    initial_state: int,
    edge_path: tuple[int, ...],
) -> tuple[
    Literal["ACCEPTING_PAIR", "INVALID_PATH"],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    str | None,
]:
    """Replay an edge path and check it is a valid accepting path.

    Returns ``(status, input_word, output_word, state_trace, error)``.
    """
    if initial_state not in transducer.initial_states:
        _reject(
            "initial_state_not_declared",
            "initial_state must select one declared initial state",
            "initial_state",
        )
    if len(edge_path) > MAX_FST_WORD_LENGTH:
        _reject(
            "edge_path_length_exceeded",
            "edge path exceeds the length bound",
            "edge_path",
        )
    if all(0 <= index < len(transducer.edges) for index in edge_path):
        input_length = sum(
            len(transducer.edges[index].input_label) for index in edge_path
        )
        output_length = sum(
            len(transducer.edges[index].output_label) for index in edge_path
        )
        if max(input_length, output_length) > MAX_FST_RESULT_WORD_LENGTH:
            _reject(
                "replay_labels_exceed_bound",
                "replayed labels exceed the result word bound",
                "edge_path",
            )
    accepting = set(transducer.accepting_states)
    if not edge_path:
        if initial_state in accepting:
            return ("ACCEPTING_PAIR", (), (), (initial_state,), None)
        return (
            "INVALID_PATH",
            (),
            (),
            (initial_state,),
            "start state not accepting",
        )
    current_state = initial_state
    state_trace: list[int] = [initial_state]
    input_word: list[int] = []
    output_word: list[int] = []
    for index in edge_path:
        if not 0 <= index < len(transducer.edges):
            return _invalid_replay(
                input_word, output_word, state_trace, f"edge index {index} out of range"
            )
        edge = transducer.edges[index]
        if edge.source != current_state:
            return _invalid_replay(
                input_word,
                output_word,
                state_trace,
                f"edge {index} source does not match current state {current_state}",
            )
        input_word.extend(edge.input_label)
        output_word.extend(edge.output_label)
        if (
            len(input_word) > MAX_FST_RESULT_WORD_LENGTH
            or len(output_word) > MAX_FST_RESULT_WORD_LENGTH
        ):
            return _invalid_replay(
                input_word[:MAX_FST_RESULT_WORD_LENGTH],
                output_word[:MAX_FST_RESULT_WORD_LENGTH],
                state_trace,
                "replayed labels exceed the result word bound",
            )
        current_state = edge.target
        state_trace.append(edge.target)
    if current_state not in accepting:
        return _invalid_replay(
            input_word, output_word, state_trace, "final state not accepting"
        )
    return (
        "ACCEPTING_PAIR",
        tuple(input_word),
        tuple(output_word),
        tuple(state_trace),
        None,
    )


def _invalid_replay(
    input_word: list[int],
    output_word: list[int],
    state_trace: list[int],
    error: str,
) -> tuple[
    Literal["INVALID_PATH"],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    str,
]:
    return (
        "INVALID_PATH",
        tuple(input_word),
        tuple(output_word),
        tuple(state_trace),
        error,
    )


def verify_subsequential_run(claim: SubseqRunResult) -> bool:
    """Verify a serialized run outcome against its transducer and word."""

    try:
        expected = run_subsequential(claim.transducer, claim.word)
        return expected == (
            claim.status,
            claim.output,
            claim.final_state,
            claim.undefined_position,
            claim.partial_output,
        )
    except (TypeError, ValueError, OperationDomainValidationError):
        return False


def verify_composition(claim: ComposeResult) -> bool:
    """Verify a candidate composite transducer against both source machines."""

    try:
        return compose_subsequential(claim.first, claim.second) == claim.transducer
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, OperationDomainValidationError):
        return False


def _minimize_sample_word_count(alphabet_size: int, max_length: int) -> int:
    """Return the number of words of length at most ``max_length``."""

    if alphabet_size <= 1:
        return max_length + 1
    count = 0
    power = 1
    for _ in range(max_length + 1):
        count += power
        if count > MAX_MINIMIZE_SAMPLE_WORDS:
            return count
        power *= alphabet_size
    return count


def _admit_minimize(
    transducer: SubsequentialTransducer, sample_max_length: int
) -> None:
    if not isinstance(transducer, SubsequentialTransducer):
        _reject(
            "transducer_type",
            "transducer must be a SubsequentialTransducer value",
            "transducer",
        )
    if type(sample_max_length) is not int or sample_max_length < 0:
        _reject(
            "sample_length",
            "sample_max_length must be a nonnegative integer",
            "sample_max_length",
        )
    words = _minimize_sample_word_count(
        transducer.input_alphabet_size, sample_max_length
    )
    if words > MAX_MINIMIZE_SAMPLE_WORDS:
        raise OperationResourceAdmissionError(
            location=("transducer", "sample_max_length"),
            code="finite_state_transducer.minimize_sample_bound_exceeded",
            message=(
                "the distinguishing sample exceeds the admitted word budget; "
                "shrink sample_max_length"
            ),
        )


def _refine_subsequential_partition(
    state_count: int,
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
) -> list[int]:
    """Refine behavioral equivalence by stable partition refinement.

    Two states share a block exactly when they realize the same partial
    function: equal final outputs and, per input symbol, jointly undefined
    or equal output words with equivalent targets.  Trimming to coaccessible
    states first makes the fixed point the coarsest such partition, so every
    separated pair is genuinely inequivalent.
    """

    block_of = [-1] * state_count
    seed_keys: dict[tuple[int, tuple[int, ...] | None], int] = {}
    for state in range(state_count):
        key = (1, finals.get(state))
        if key not in seed_keys:
            seed_keys[key] = len(seed_keys)
        block_of[state] = seed_keys[key]
    while True:
        split_found = False
        next_block = [-1] * state_count
        next_id = 0
        for _block in sorted(set(block_of)):
            members = sorted(
                state for state in range(state_count) if block_of[state] == _block
            )
            signatures: dict[
                tuple[tuple[tuple[int, ...], tuple[int]] | None, ...], list[int]
            ] = {}
            for state in members:
                signature: list[tuple[tuple[int, ...], tuple[int]] | None] = []
                for symbol in range(alphabet_size):
                    target = transitions.get((state, symbol))
                    if target is None:
                        signature.append(None)
                    else:
                        next_state, output = target
                        signature.append((output, (block_of[next_state],)))
                signature_key = tuple(signature)
                signatures.setdefault(signature_key, []).append(state)
            groups = list(signatures.values())
            if len(groups) > 1:
                split_found = True
            for group in groups:
                for state in group:
                    next_block[state] = next_id
                next_id += 1
        block_of = next_block
        if not split_found:
            return block_of


_MAX_WITNESS_NODES = 200_000
"""Bound on pair nodes explored per inequivalent pair before refusing.

The search tracks the residual output difference between the two states,
which the admitted envelopes keep small; the cap only guards a pathological
output-heavy request from unbounded work.
"""


def _advance_output_residual(
    sign: int,
    residual: tuple[int, ...],
    first_output: tuple[int, ...],
    second_output: tuple[int, ...],
) -> tuple[int, tuple[int, ...]] | None:
    """Fold one emitted-output pair into the residual difference.

    ``sign`` is ``0`` when the accumulated outputs are equal, ``1`` when the
    first state is ahead by ``residual``, and ``-1`` when the second is.
    Returns the new ``(sign, residual)`` with the common prefix stripped when
    the accumulated outputs stay prefix-comparable, or ``None`` when they are
    incomparable (a difference no continuation can cancel).
    """

    if sign == 0:
        left, right = first_output, second_output
    elif sign == 1:
        left, right = residual + first_output, second_output
    else:
        left, right = first_output, residual + second_output
    common = 0
    while common < len(left) and common < len(right) and left[common] == right[common]:
        common += 1
    left = left[common:]
    right = right[common:]
    if not left and not right:
        return (0, ())
    if not left:
        return (-1, right)
    if not right:
        return (1, left)
    return None


def _witness_terminal(
    node: tuple[int | None, int | None, int, tuple[int, ...]],
    finals: dict[int, tuple[int, ...]],
) -> bool:
    """Decide whether a pair node realizes differing partial functions."""

    first, second, sign, residual = node
    first_final = first is not None and first in finals
    second_final = second is not None and second in finals
    if first_final != second_final:
        return True
    if not (first_final and second_final):
        return False
    if first is None or second is None:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    first_output = finals[first]
    second_output = finals[second]
    if sign == 2:
        return True
    if sign == 0:
        return first_output != second_output
    if sign == 1:
        return residual + first_output != second_output
    return first_output != residual + second_output


def _find_separating_word(
    first: int,
    second: int,
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Breadth-first search for the shortest word separating two states.

    Nodes track the residual output difference so a transition-level
    difference canceled by later output or a final output is not mistaken
    for a separation.  The residual is exact; the node budget bounds the
    search instead of a length cap, since a residual may be canceled by any
    later emitted output.
    """

    start: tuple[int | None, int | None, int, tuple[int, ...]] = (
        first,
        second,
        0,
        (),
    )
    if _witness_terminal(start, finals):
        return ()
    seen = {start}
    queue: deque[
        tuple[tuple[int | None, int | None, int, tuple[int, ...]], tuple[int, ...]]
    ] = deque([(start, ())])
    explored = 0
    while queue:
        node, word = queue.popleft()
        explored += 1
        if explored > _MAX_WITNESS_NODES:
            raise OperationResourceAdmissionError(
                location=("transducer",),
                code="finite_state_transducer.witness_search_exceeded",
                message=("the distinguishing-witness search exceeds its node budget"),
            )
        current_first, current_second, sign, residual = node
        for symbol in range(alphabet_size):
            first_step = (
                transitions.get((current_first, symbol))
                if current_first is not None
                else None
            )
            second_step = (
                transitions.get((current_second, symbol))
                if current_second is not None
                else None
            )
            if first_step is None and second_step is None:
                continue
            next_first = first_step[0] if first_step is not None else None
            next_second = second_step[0] if second_step is not None else None
            next_sign = sign
            next_residual = residual
            if sign != 2:
                advanced = _advance_output_residual(
                    sign,
                    residual,
                    first_step[1] if first_step is not None else (),
                    second_step[1] if second_step is not None else (),
                )
                if advanced is None:
                    next_sign, next_residual = 2, ()
                else:
                    next_sign, next_residual = advanced
            child = (next_first, next_second, next_sign, next_residual)
            extended = (*word, symbol)
            if _witness_terminal(child, finals):
                return extended
            if child not in seen:
                seen.add(child)
                queue.append((child, extended))
    return None


def _compute_distinguishing_witnesses(
    state_count: int,
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
    block_of: list[int],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Return a separating word for every pair of inequivalent states.

    Witnesses are computed only after the partition has stabilized, so a
    shared block means the two states realize the same partial function and
    is skipped.  Each inequivalent pair is witnessed by the shortest word
    whose realized partial functions differ.
    """

    witnesses: dict[tuple[int, int], tuple[int, ...]] = {}
    for first in range(state_count):
        for second in range(first + 1, state_count):
            if block_of[first] == block_of[second]:
                continue
            word = _find_separating_word(
                first,
                second,
                alphabet_size,
                transitions,
                finals,
            )
            if word is None:
                raise RuntimeError(
                    "partition refinement separated pairs without distinguishing words"
                )
            witnesses[(first, second)] = word
    return witnesses


def _iter_sample_words(
    alphabet_size: int, max_length: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every word of length at most ``max_length`` in lex order."""

    words: list[tuple[int, ...]] = [()]
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(max_length):
        following: list[tuple[int, ...]] = []
        for word in frontier:
            for symbol in range(alphabet_size):
                extended = (*word, symbol)
                words.append(extended)
                following.append(extended)
        frontier = following
    return tuple(words)


def _is_common_output_prefix(
    state: int,
    prefix: tuple[int, ...],
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
) -> bool:
    """Decide whether every accepted output from ``state`` starts with ``prefix``.

    A breadth-first search over ``(state, matched)`` configurations finds a
    counterexample path: a transition whose output mismatches the next prefix
    symbol, or an accepting stop whose output is shorter (or mismatched)
    before the prefix is matched.  Once ``matched`` reaches the prefix length
    the remaining output is unconstrained, so that branch stops.
    """

    target = len(prefix)
    visited: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = [(state, 0)]
    while stack:
        current, matched = stack.pop()
        if matched >= target or (current, matched) in visited:
            continue
        visited.add((current, matched))
        final = finals.get(current)
        if final is not None:
            span = min(len(final), target - matched)
            if final[:span] != prefix[matched : matched + span]:
                return False
            if matched + len(final) < target:
                return False
        for symbol in range(alphabet_size):
            step = transitions.get((current, symbol))
            if step is None:
                continue
            target_state, output = step
            span = min(len(output), target - matched)
            if output[:span] != prefix[matched : matched + span]:
                return False
            stack.append((target_state, matched + span))
    return True


def _output_prefixes(
    state_count: int,
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
    initial_state: int,
) -> list[tuple[int, ...]]:
    """Return each state's longest common output prefix (initial pinned empty).

    The prefix of a state is the longest word shared by the outputs of every
    successful path starting there, including the reached final outputs.  It
    is grown one symbol at a time: at most one symbol can extend the current
    common prefix, so the greedy extension is exact.  The initial state is
    pinned to the empty prefix because this value model has no initial output
    to absorb a global prefix; pinning it keeps the realized function
    unchanged.
    """

    prefix: list[tuple[int, ...]] = []
    for state in range(state_count):
        current: tuple[int, ...] = ()
        while True:
            extended = next(
                (
                    (*current, symbol)
                    for symbol in range(alphabet_size)
                    if _is_common_output_prefix(
                        state,
                        (*current, symbol),
                        alphabet_size,
                        transitions,
                        finals,
                    )
                ),
                None,
            )
            if extended is None:
                break
            current = extended
        prefix.append(current)
    prefix[initial_state] = ()
    return prefix


def _push_output_prefixes(
    state_count: int,
    alphabet_size: int,
    transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    finals: dict[int, tuple[int, ...]],
    initial_state: int,
) -> tuple[
    dict[tuple[int, int], tuple[int, tuple[int, ...]]],
    dict[int, tuple[int, ...]],
]:
    """Rewrite outputs so no live state carries a common output prefix.

    Each transition ``q --s/w--> q'`` becomes ``q --s/u--> q'`` with
    ``c(q) u = w c(q')``, and each final output ``f`` becomes
    ``strip(c(q), f)``.  The telescoping identity makes the total output of
    every accepted path unchanged, so the realized partial function is
    preserved while the output prefixes move to the incoming transitions.
    """

    prefix = _output_prefixes(
        state_count, alphabet_size, transitions, finals, initial_state
    )
    pushed_transitions: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for (state, symbol), (target, output) in transitions.items():
        combined = output + prefix[target]
        head = prefix[state]
        if combined[: len(head)] != head:
            raise RuntimeError(
                "output-prefix normalization broke a transition prefix invariant"
            )
        pushed_transitions[(state, symbol)] = (target, combined[len(head) :])
    pushed_finals: dict[int, tuple[int, ...]] = {}
    for state, final in finals.items():
        head = prefix[state]
        if final[: len(head)] != head:
            raise RuntimeError(
                "output-prefix normalization broke a final-output prefix invariant"
            )
        pushed_finals[state] = final[len(head) :]
    return pushed_transitions, pushed_finals


def minimize_subsequential(
    transducer: SubsequentialTransducer,
    sample_max_length: int = 5,
) -> MinimizeResult:
    """Minimize a subsequential transducer preserving its partial function.

    The kernel trims to live states, pushes common output prefixes onto the
    incoming transitions so no live state carries a redundant output prefix,
    merges the coarsest exact-output bisimulation of the normalized machine
    by partition refinement, and replays the shipped run semantics on every
    word of length at most ``sample_max_length``.  Pushing preserves the
    realized partial function and can only merge states the raw exact-output
    bisimulation would separate, so the quotient is the state-minimal machine
    for this value model.
    """

    _admit_minimize(transducer, sample_max_length)
    trimmed, trim_map = trim_subsequential(transducer)
    trim_new_to_old = {new: old for old, new in trim_map.items()}
    transitions = _transition_map(trimmed)
    finals = _final_output_map(trimmed)
    if trimmed.state_count == 1 and not trim_map:
        return MinimizeResult._from_kernel(
            transducer=transducer,
            sample_max_length=sample_max_length,
            minimized=trimmed,
            old_to_new=tuple(-1 for _ in range(transducer.state_count)),
            new_to_old=(),
            partition=(),
            distinguishability=(),
            sample_words_checked=_minimize_sample_word_count(
                transducer.input_alphabet_size, sample_max_length
            ),
            sample_agreement=True,
        )
    transitions, finals = _push_output_prefixes(
        trimmed.state_count,
        trimmed.input_alphabet_size,
        transitions,
        finals,
        trimmed.initial_state,
    )
    block_of = _refine_subsequential_partition(
        trimmed.state_count,
        trimmed.input_alphabet_size,
        transitions,
        finals,
    )
    witnesses = _compute_distinguishing_witnesses(
        trimmed.state_count,
        trimmed.input_alphabet_size,
        transitions,
        finals,
        block_of,
    )
    blocks: dict[int, list[int]] = {}
    for state, block in enumerate(block_of):
        blocks.setdefault(block, []).append(state)
    ordered = sorted(blocks.values(), key=min)
    state_to_block = {
        state: index for index, block in enumerate(ordered) for state in block
    }
    rep_of = {index: min(block) for index, block in enumerate(ordered)}
    minimized_transitions = tuple(
        sorted(
            (
                SubseqTransition(
                    source=state_to_block[rep],
                    input_symbol=symbol,
                    target=state_to_block[target],
                    output=output,
                )
                for rep in (rep_of[index] for index in range(len(ordered)))
                for symbol in range(trimmed.input_alphabet_size)
                if (rep, symbol) in transitions
                for (target, output) in [transitions[(rep, symbol)]]
            ),
            key=lambda row: (row.source, row.input_symbol, row.target, row.output),
        )
    )
    minimized_finals = tuple(
        sorted(
            (
                SubseqFinalOutput(state=index, output=finals[rep_of[index]])
                for index in range(len(ordered))
                if rep_of[index] in finals
            ),
            key=lambda row: row.state,
        )
    )
    minimized = SubsequentialTransducer(
        input_alphabet_size=trimmed.input_alphabet_size,
        output_alphabet_size=trimmed.output_alphabet_size,
        state_count=len(ordered),
        initial_state=state_to_block[trim_map[transducer.initial_state]],
        transitions=minimized_transitions,
        final_outputs=minimized_finals,
    )
    sample_words = _iter_sample_words(transducer.input_alphabet_size, sample_max_length)
    for word in sample_words:
        source_status, source_output, _, _, _ = run_subsequential(transducer, word)
        minimized_status, minimized_output, _, _, _ = run_subsequential(minimized, word)
        # The shipped trim/run semantics preserve the realized partial
        # function: definedness with equal output words. Distinct failure
        # modes (undefined transition vs nonfinal state) both mean the word
        # is outside the domain.
        if (source_status == "OUTPUT", source_output) != (
            minimized_status == "OUTPUT",
            minimized_output,
        ):
            raise RuntimeError("minimized transducer disagrees with its source")
    old_to_new = tuple(
        state_to_block[trim_map[state]] if state in trim_map else -1
        for state in range(transducer.state_count)
    )
    new_to_old = tuple(trim_new_to_old[rep_of[index]] for index in range(len(ordered)))
    partition = tuple(
        tuple(sorted(trim_new_to_old[state] for state in block)) for block in ordered
    )
    original_of = {new: trim_new_to_old[new] for new in range(trimmed.state_count)}
    outcomes: list[_StatePairOutcome] = []
    for first in range(trimmed.state_count):
        for second in range(first + 1, trimmed.state_count):
            if block_of[first] == block_of[second]:
                outcomes.append(
                    _EquivalentStatePair(
                        first_state=original_of[first],
                        second_state=original_of[second],
                    )
                )
            else:
                if (first, second) not in witnesses:
                    raise RuntimeError(
                        "partition refinement split a pair without a witness"
                    )
                outcomes.append(
                    _SeparatedStatePair(
                        first_state=original_of[first],
                        second_state=original_of[second],
                        witness_word=witnesses[(first, second)],
                    )
                )
    table_rows = [
        StatePairDistinguishability(
            first_state=outcome.first_state,
            second_state=outcome.second_state,
            equivalent=isinstance(outcome, _EquivalentStatePair),
            witness_word=(
                ()
                if isinstance(outcome, _EquivalentStatePair)
                else outcome.witness_word
            ),
        )
        for outcome in outcomes
    ]
    table_rows.sort(key=lambda row: (row.first_state, row.second_state))
    distinguishability = tuple(table_rows)
    return MinimizeResult._from_kernel(
        transducer=transducer,
        sample_max_length=sample_max_length,
        minimized=minimized,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        partition=partition,
        distinguishability=distinguishability,
        sample_words_checked=len(sample_words),
        sample_agreement=True,
    )


def verify_minimization(claim: MinimizeResult) -> bool:
    """Verify a minimization against its retained source transducer."""

    try:
        return (
            minimize_subsequential(claim.transducer, claim.sample_max_length) == claim
        )
    except OperationResourceAdmissionError:
        raise
    except (TypeError, ValueError, OperationDomainValidationError):
        return False
