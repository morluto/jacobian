"""Regular language operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.regular_languages._models import (
    ComplementRequest,
    ComplementResult,
    CountRequest,
    CountResult,
    RunRequest,
    RunResult,
)
from jacobian.math.regular_languages._operations import (
    compute_complement,
    compute_count,
    compute_run,
)


def rl_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
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


REGULAR_LANGUAGE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    rl_operation(
        "regular_language.run.check",
        "Check if a word is accepted by a DFA",
        "Simulate a deterministic finite automaton on a word and return "
        "whether it is accepted and the final state reached.",
        RunRequest,
        RunResult,
        compute_run,
        "automata",
        "dfa",
        "exact",
        examples=(
            example(
                "binary_ends_in_1",
                "DFA accepting binary strings ending in 1, word [1,0,1] accepted.",
                {"dfa": _DFA_EXAMPLE["dfa"], "word": [1, 0, 1]},
            ),
        ),
    ),
    rl_operation(
        "regular_language.count_words.compute",
        "Count accepted words of a given length",
        "Count the number of words of exact length accepted by a DFA "
        "using exact integer matrix powering of the transition matrix.",
        CountRequest,
        CountResult,
        compute_count,
        "automata",
        "counting",
        "exact",
        examples=(
            example(
                "binary_ends_in_1",
                "Count binary strings of length 3 ending in 1: 4 words.",
                {"dfa": _DFA_EXAMPLE["dfa"], "word_length": 3},
            ),
        ),
    ),
    rl_operation(
        "regular_language.complement.compute",
        "Compute the complement of a DFA's language",
        "Compute the complement DFA by flipping the set of accepting states.",
        ComplementRequest,
        ComplementResult,
        compute_complement,
        "automata",
        "complement",
        "exact",
        examples=(
            example(
                "binary_ends_in_1",
                "Complement of DFA accepting strings ending in 1.",
                {"dfa": _DFA_EXAMPLE["dfa"]},
            ),
        ),
    ),
)

TOOLS = REGULAR_LANGUAGE_OPERATIONS

__all__ = ["TOOLS"]
