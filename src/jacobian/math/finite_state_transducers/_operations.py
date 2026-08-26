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


def verify_subseq_run_result(result: SubseqRunResult) -> bool:
    """Replay one independently supplied subsequential-run claim."""

    expected = run_subsequential(result.transducer, result.word)
    return (
        result.status,
        result.output,
        result.final_state,
        result.undefined_position,
        result.partial_output,
    ) == expected


def verify_compose_result(result: ComposeResult) -> bool:
    """Replay one independently supplied bounded composition claim."""

    return result.transducer == compose_subsequential(result.first, result.second)


def verify_relation_path_replay_result(result: RelationPathReplayResult) -> bool:
    """Replay one independently supplied rational-relation path claim."""

    return (
        result.status,
        result.input_word,
        result.output_word,
        result.state_trace,
        result.error,
    ) == replay_rational_path(result.transducer, result.initial_state, result.edge_path)


__all__ = [
    "compute_compose",
    "compute_relation_path_replay",
    "compute_run",
    "verify_compose_result",
    "verify_relation_path_replay_result",
    "verify_subseq_run_result",
]
