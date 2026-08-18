"""Wire adapters for exact bounded finite-state transducers."""

from __future__ import annotations

from jacobian.math.finite_state_transducers._models import (
    ComposeRequest,
    ComposeResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
)
from jacobian.math.finite_state_transducers.operations import (
    compose_subsequential,
    replay_rational_path,
    run_subsequential,
)


def compute_run(request: SubseqRunRequest) -> SubseqRunResult:
    status, output, final_state, undefined_position, partial_output = run_subsequential(
        request.transducer, request.word
    )
    return SubseqRunResult(
        transducer=request.transducer,
        word=request.word,
        status=status,
        output=output,
        final_state=final_state,
        undefined_position=undefined_position,
        partial_output=partial_output,
    )


def compute_compose(request: ComposeRequest) -> ComposeResult:
    return ComposeResult(
        first=request.first,
        second=request.second,
        transducer=compose_subsequential(request.first, request.second),
    )


def compute_relation_path_replay(
    request: RelationPathReplayRequest,
) -> RelationPathReplayResult:
    status, input_word, output_word, state_trace, error = replay_rational_path(
        request.transducer, request.initial_state, request.edge_path
    )
    return RelationPathReplayResult(
        transducer=request.transducer,
        initial_state=request.initial_state,
        edge_path=request.edge_path,
        status=status,
        input_word=input_word,
        output_word=output_word,
        state_trace=state_trace,
        error=error,
    )


__all__ = [
    "compute_compose",
    "compute_relation_path_replay",
    "compute_run",
]
