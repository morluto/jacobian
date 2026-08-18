"""Domain adapter for finite-state transducer operations."""

from __future__ import annotations

from jacobian.math.finite_state_transducers._models import (
    ComposeRequest,
    ComposeResult,
    IdentityRequest,
    IdentityResult,
    RelationInverseRequest,
    RelationInverseResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
    TrimRequest,
    TrimResult,
)
from jacobian.math.finite_state_transducers.operations import (
    compose_subsequential,
    identity_transducer,
    invert_rational,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
)

__all__ = [
    "compute_compose",
    "compute_identity",
    "compute_relation_inverse",
    "compute_relation_path_replay",
    "compute_run",
    "compute_trim",
]


def compute_run(request: SubseqRunRequest) -> SubseqRunResult:
    status, output, final_state, undef_pos, partial = run_subsequential(
        request.transducer, request.word
    )
    if status == "OUTPUT":
        return SubseqRunResult(
            status="OUTPUT",
            output=output,
            final_state=final_state,
            undefined_position=None,
            partial_output=(),
        )
    if status == "UNDEFINED_TRANSITION":
        return SubseqRunResult(
            status="UNDEFINED_TRANSITION",
            output=(),
            final_state=final_state,
            undefined_position=undef_pos,
            partial_output=partial,
        )
    return SubseqRunResult(
        status="NONFINAL_DOMAIN_STATE",
        output=(),
        final_state=final_state,
        undefined_position=None,
        partial_output=partial,
    )


def compute_identity(request: IdentityRequest) -> IdentityResult:
    return IdentityResult(transducer=identity_transducer(request.alphabet_size))


def compute_compose(request: ComposeRequest) -> ComposeResult:
    return ComposeResult(
        transducer=compose_subsequential(request.first, request.second)
    )


def compute_trim(request: TrimRequest) -> TrimResult:
    transducer, state_map = trim_subsequential(request.transducer)
    return TrimResult(transducer=transducer, state_map=state_map)


def compute_relation_inverse(
    request: RelationInverseRequest,
) -> RelationInverseResult:
    return RelationInverseResult(transducer=invert_rational(request.transducer))


def compute_relation_path_replay(
    request: RelationPathReplayRequest,
) -> RelationPathReplayResult:
    status, input_word, output_word, state_trace, error = replay_rational_path(
        request.transducer, request.edge_path
    )
    if status == "ACCEPTING_PAIR":
        return RelationPathReplayResult(
            status="ACCEPTING_PAIR",
            input_word=input_word,
            output_word=output_word,
            state_trace=state_trace,
            error=None,
        )
    return RelationPathReplayResult(
        status="INVALID_PATH",
        input_word=input_word,
        output_word=output_word,
        state_trace=state_trace,
        error=error,
    )
