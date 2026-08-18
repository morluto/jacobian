"""Domain-owned finite-state transducer kernels."""

from __future__ import annotations

from collections import deque

from jacobian.math.finite_state_transducers.values import (
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)

__all__ = [
    "coaccessible_states",
    "compose_subsequential",
    "identity_transducer",
    "invert_rational",
    "reachable_states",
    "replay_rational_path",
    "run_subsequential",
    "trim_subsequential",
]


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
) -> tuple[str, tuple[int, ...], int, int | None, tuple[int, ...]]:
    """Run a subsequential transducer on ``word``.

    Returns ``(status, output, final_state, undefined_position, partial_output)``.

    ``status`` is one of ``"OUTPUT"``, ``"UNDEFINED_TRANSITION"``, or
    ``"NONFINAL_DOMAIN_STATE"``.
    """
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


def trim_subsequential(
    transducer: SubsequentialTransducer,
) -> tuple[SubsequentialTransducer, dict[int, int]]:
    """Restrict a transducer to reachable and coaccessible states.

    Returns the trimmed transducer and an old-state -> new-state map.
    """
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
    """
    t_map = {
        (tr.source, tr.input_symbol): (tr.target, tr.output)
        for tr in first.transitions
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
            pair_next = (t_next, u_next_final)
            if pair_next not in state_pairs:
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

    new_finals: list[SubseqFinalOutput] = []
    for (t_state, u_state), new_state in state_pairs.items():
        if t_state not in t_finals:
            continue
        t_final_word = t_finals[t_state]
        result = _run_u_on_word(u_map, u_state, t_final_word)
        if result is None:
            continue
        u_final_state, composite_output = result
        if u_final_state in u_finals:
            composite_output.extend(u_finals[u_final_state])
        new_finals.append(
            SubseqFinalOutput(state=new_state, output=tuple(composite_output))
        )

    return SubsequentialTransducer(
        input_alphabet_size=first.input_alphabet_size,
        output_alphabet_size=second.output_alphabet_size,
        state_count=len(state_pairs),
        initial_state=0,
        transitions=tuple(new_transitions),
        final_outputs=tuple(new_finals),
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


def replay_rational_path(  # noqa: C901
    transducer: RationalTransducer,
    edge_path: tuple[int, ...],
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...], str | None]:
    """Replay an edge path and check it is a valid accepting path.

    Returns ``(status, input_word, output_word, state_trace, error)``.
    """
    if not edge_path:
        if not transducer.initial_states:
            return ("INVALID_PATH", (), (), (), "no initial states")
        return (
            "ACCEPTING_PAIR" if transducer.initial_states[0] in set(
                transducer.accepting_states
            )
            else "INVALID_PATH",
            (),
            (),
            (transducer.initial_states[0],),
            None if transducer.initial_states[0] in set(
                transducer.accepting_states
            )
            else "start state not accepting",
        )
    edges = transducer.edges
    state_trace: list[int] = []
    input_word: list[int] = []
    output_word: list[int] = []
    for idx in edge_path:
        if idx < 0 or idx >= len(edges):
            return ("INVALID_PATH", (), (), (), f"edge index {idx} out of range")
        if not state_trace:
            if edges[edge_path[0]].source not in set(transducer.initial_states):
                return (
                    "INVALID_PATH",
                    (),
                    (),
                    (),
                    "first edge source is not an initial state",
                )
            state_trace.append(edges[edge_path[0]].source)
    for i, idx in enumerate(edge_path):
        edge = edges[idx]
        if i > 0:
            prev_target = edges[edge_path[i - 1]].target
            if edge.source != prev_target:
                return (
                    "INVALID_PATH",
                    (),
                    (),
                    (),
                    f"edge {idx} source {edge.source} != prev target {prev_target}",
                )
        input_word.extend(edge.input_label)
        output_word.extend(edge.output_label)
        state_trace.append(edge.target)
    last_state = state_trace[-1]
    if last_state not in set(transducer.accepting_states):
        return (
            "INVALID_PATH",
            tuple(input_word),
            tuple(output_word),
            tuple(state_trace),
            "final state not accepting",
        )
    return (
        "ACCEPTING_PAIR",
        tuple(input_word),
        tuple(output_word),
        tuple(state_trace),
        None,
    )
