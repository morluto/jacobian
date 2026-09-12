"""Finite-state transducer operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers._models import (
    ComposeRequest,
    ComposeResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
)
from jacobian.math.logic.automata.transducers.operations import (
    compose_subsequential,
    replay_rational_path,
    run_subsequential,
)


def compute_run(request: SubseqRunRequest) -> SubseqRunResult:
    status, output, final_state, undefined_position, partial_output = run_subsequential(
        request.transducer, request.word
    )
    return SubseqRunResult._from_kernel(
        request,
        status=status,
        output=output,
        final_state=final_state,
        undefined_position=undefined_position,
        partial_output=partial_output,
    )


def compute_compose(request: ComposeRequest) -> ComposeResult:
    return ComposeResult._from_kernel(
        request,
        transducer=compose_subsequential(request.first, request.second),
    )


def compute_relation_path_replay(
    request: RelationPathReplayRequest,
) -> RelationPathReplayResult:
    status, input_word, output_word, state_trace, error = replay_rational_path(
        request.transducer, request.initial_state, request.edge_path
    )
    return RelationPathReplayResult._from_kernel(
        request,
        status=status,
        input_word=input_word,
        output_word=output_word,
        state_trace=state_trace,
        error=error,
    )


_IDENTITY = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 2,
    "state_count": 1,
    "initial_state": 0,
    "transitions": [
        {"source": 0, "input_symbol": 0, "target": 0, "output": [0]},
        {"source": 0, "input_symbol": 1, "target": 0, "output": [1]},
    ],
    "final_outputs": [{"state": 0, "output": []}],
}

_FLIP = {
    **_IDENTITY,
    "transitions": [
        {"source": 0, "input_symbol": 0, "target": 0, "output": [1]},
        {"source": 0, "input_symbol": 1, "target": 0, "output": [0]},
    ],
}

_RELATION = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 2,
    "state_count": 1,
    "initial_states": [0],
    "accepting_states": [0],
    "edges": [
        {"source": 0, "target": 0, "input_label": [0], "output_label": [1]},
        {"source": 0, "target": 0, "input_label": [1], "output_label": [0]},
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="transducer.subsequential.run.compute",
        title="Run a subsequential transducer on a word",
        description="Execute one exact bounded run, distinguishing successful empty output, "
        "an undefined transition, and termination in a nonfinal state.",
        request_type=SubseqRunRequest,
        result_type=SubseqRunResult,
        run=compute_run,
        tags=("transducer", "subsequential", "exact"),
        examples=(
            OperationExample(
                name="binary_identity_run",
                description="Run the binary identity transducer on a three-symbol word.",
                input={"transducer": _IDENTITY, "word": [0, 1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.subsequential.compose.compute",
        title="Compose two subsequential transducers",
        description="Construct the exact bounded subsequential transducer for U after T, "
        "including both transition and final-output domain restrictions.",
        request_type=ComposeRequest,
        result_type=ComposeResult,
        run=compute_compose,
        tags=("transducer", "subsequential", "composition", "exact"),
        examples=(
            OperationExample(
                name="identity_then_flip",
                description="Compose the binary identity with the binary symbol flip.",
                input={"first": _IDENTITY, "second": _FLIP},
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.relation.path.replay.compute",
        title="Replay a rational-relation path",
        description="Replay one candidate edge-index path from an explicitly selected "
        "initial state and return its exact labels, trace, and acceptance status.",
        request_type=RelationPathReplayRequest,
        result_type=RelationPathReplayResult,
        run=compute_relation_path_replay,
        tags=("transducer", "rational-relation", "path-replay", "exact"),
        examples=(
            OperationExample(
                name="two_edge_bit_flip_path",
                description="Replay two edges from initial state zero.",
                input={
                    "transducer": _RELATION,
                    "initial_state": 0,
                    "edge_path": [0, 1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
