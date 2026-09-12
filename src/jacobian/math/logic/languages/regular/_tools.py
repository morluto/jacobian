"""Regular language operation declarations."""

from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.logic.languages.regular import (
    TransitionParikhProfile,
    count_accepted_words,
    dfa_complement,
    dfa_equivalence,
    dfa_run,
    transition_parikh_profile,
)
from jacobian.math.logic.languages.regular._models import (
    ComplementRequest,
    ComplementResult,
    CountRequest,
    CountResult,
    EquivalenceRequest,
    EquivalenceResult,
    RunRequest,
    RunResult,
    TransitionParikhProfileRequest,
)
from jacobian.math.logic.languages.regular._symbol_parikh_tools import (
    SYMBOL_PARIKH_PROFILE_OPERATION,
)


def compute_run(request: RunRequest) -> RunResult:
    try:
        accepted, final_state = dfa_run(request.dfa, request.word)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("word",),
            code="regular_language.word_symbol_out_of_range",
            message=str(exc),
        ) from exc
    transitions = {
        (item.source, item.symbol): item.target for item in request.dfa.transitions
    }
    trace = [request.dfa.initial_state]
    for symbol in request.word:
        trace.append(transitions[(trace[-1], symbol)])
    return RunResult._from_kernel(
        request,
        accepted=accepted,
        final_state=final_state,
        state_trace=tuple(trace),
    )


def compute_count(request: CountRequest) -> CountResult:
    count = count_accepted_words(request.dfa, request.word_length)
    return CountResult._from_kernel(
        request,
        count=count,
    )


def compute_complement(request: ComplementRequest) -> ComplementResult:
    return ComplementResult(dfa=dfa_complement(request.dfa))


def compute_equivalence(request: EquivalenceRequest) -> EquivalenceResult:
    word, left_trace, right_trace = dfa_equivalence(request.left, request.right)
    request_checkpoint("before DFA equivalence result construction")
    return EquivalenceResult(
        left=request.left,
        right=request.right,
        equivalent=word is None,
        distinguishing_word=word,
        left_state_trace=left_trace,
        right_state_trace=right_trace,
    )


def compute_transition_parikh_profile(
    request: TransitionParikhProfileRequest,
) -> TransitionParikhProfile:
    return transition_parikh_profile(
        request.automaton,
        request.source_state,
        request.target_state,
        request.path_length,
    )


_DFA_EXAMPLE = {
    "dfa": {
        "state_count": 2,
        "alphabet_size": 2,
        "transitions": [
            {"source": 0, "symbol": 0, "target": 0},
            {"source": 0, "symbol": 1, "target": 1},
            {"source": 1, "symbol": 0, "target": 0},
            {"source": 1, "symbol": 1, "target": 1},
        ],
        "initial_state": 0,
        "accepting_states": [1],
    },
}

_TRANSITION_PROFILE_EXAMPLE = {
    "automaton": {
        "state_count": 1,
        "alphabet_size": 2,
        "transitions": [
            {"transition_id": 0, "source": 0, "symbol": 0, "target": 0},
            {"transition_id": 1, "source": 0, "symbol": 1, "target": 0},
        ],
    },
    "source_state": 0,
    "target_state": 0,
    "path_length": 2,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    SYMBOL_PARIKH_PROFILE_OPERATION,
    MathTool(
        operation_id="regular_language.dfa.equivalence.decide",
        title="Decide deterministic finite automaton equivalence",
        description=(
            "Decide whether two total DFAs over the same ordered alphabet accept "
            "the same language by reachable-product search. Inequivalence returns "
            "the shortest lexicographically least distinguishing word and both run traces."
        ),
        request_type=EquivalenceRequest,
        result_type=EquivalenceResult,
        run=compute_equivalence,
        tags=("regular-language", "automata", "dfa", "equivalence", "exact"),
        examples=(
            OperationExample(
                name="different_empty_word_acceptance",
                description=(
                    "Decide that these two total DFAs differ and return the empty-word "
                    "witness; both must share one ordered alphabet and a complete "
                    "state-symbol transition table."
                ),
                input={
                    "left": _DFA_EXAMPLE["dfa"],
                    "right": {**_DFA_EXAMPLE["dfa"], "accepting_states": [0]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="regular_language.complement.compute",
        title="Complement a deterministic finite automaton",
        description="Return the complete DFA over the same alphabet with accepting and "
        "non-accepting states exchanged.",
        request_type=ComplementRequest,
        result_type=ComplementResult,
        run=compute_complement,
        tags=("automata", "dfa", "complement", "exact"),
        examples=(
            OperationExample(
                name="binary_ends_in_1_complement",
                description="Complement the DFA accepting binary strings ending in one.",
                input={"dfa": _DFA_EXAMPLE["dfa"]},
            ),
        ),
    ),
    MathTool(
        operation_id="automaton.path.transition_parikh_profile.compute",
        title="Compute a transition-Parikh profile for fixed-length automaton paths",
        description="Return the complete exact histogram from transition-use vectors to path "
        "multiplicities for one source, target, and exact length. Coordinates use "
        "the automaton's stable transition-ID axis; requests above the derived "
        "work or result envelope are rejected before the sparse recurrence runs.",
        request_type=TransitionParikhProfileRequest,
        result_type=TransitionParikhProfile,
        run=compute_transition_parikh_profile,
        tags=("automata", "paths", "parikh", "exact", "complete"),
        examples=(
            OperationExample(
                name="two_loop_transition_histogram",
                description="Compute all length-two loop paths and group them by transition "
                "counts; transition IDs must be the contiguous ordered axis.",
                input=_TRANSITION_PROFILE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="regular_language.run.check",
        title="Check if a word is accepted by a DFA",
        description="Simulate a deterministic finite automaton on a word and return "
        "whether it is accepted and the final state reached.",
        request_type=RunRequest,
        result_type=RunResult,
        run=compute_run,
        tags=("automata", "dfa", "exact"),
        examples=(
            OperationExample(
                name="binary_ends_in_1",
                description="DFA accepting binary strings ending in 1, word [1,0,1] accepted.",
                input={"dfa": _DFA_EXAMPLE["dfa"], "word": [1, 0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="regular_language.count_words.compute",
        title="Count accepted words of a given length",
        description="Count the number of words of exact length accepted by a DFA "
        "using exact integer matrix powering of the transition matrix.",
        request_type=CountRequest,
        result_type=CountResult,
        run=compute_count,
        tags=("automata", "counting", "exact"),
        examples=(
            OperationExample(
                name="binary_ends_in_1",
                description="Count binary strings of length 3 ending in 1: 4 words.",
                input={"dfa": _DFA_EXAMPLE["dfa"], "word_length": 3},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
