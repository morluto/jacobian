"""Finite-state transducer operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers._models import (
    ComposeRequest,
    ComposeResult,
    MinimizeRequest,
    MinimizeResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
    TrimRequest,
    TrimResult,
)
from jacobian.math.logic.automata.transducers.operations import (
    compose_subsequential,
    minimize_subsequential,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
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


def compute_trim(request: TrimRequest) -> TrimResult:
    trimmed, old_to_new = trim_subsequential(request.transducer)
    return TrimResult._from_kernel(request, trimmed=trimmed, old_to_new=old_to_new)


def compute_minimize(request: MinimizeRequest) -> MinimizeResult:
    return minimize_subsequential(request.transducer, request.sample_max_length)


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

# Three-state source with one reachable dead state (1) and one unreachable
# state (2). Only state 0 can reach the final output, so trim keeps {0}.
_TRIM_SOURCE = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 2,
    "state_count": 3,
    "initial_state": 0,
    "transitions": [
        {"source": 0, "input_symbol": 0, "target": 0, "output": [0]},
        {"source": 0, "input_symbol": 1, "target": 1, "output": [1]},
        {"source": 1, "input_symbol": 0, "target": 1, "output": [0]},
        {"source": 1, "input_symbol": 1, "target": 1, "output": [1]},
        {"source": 2, "input_symbol": 0, "target": 2, "output": [0]},
        {"source": 2, "input_symbol": 1, "target": 2, "output": [1]},
    ],
    "final_outputs": [{"state": 0, "output": []}],
}

# Three-state source with two equivalent live states (0 and 1): both are
# final with empty output and agree on every transition up to renaming.
_MINIMIZE_SOURCE = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 2,
    "state_count": 3,
    "initial_state": 0,
    "transitions": [
        {"source": 0, "input_symbol": 0, "target": 1, "output": [0]},
        {"source": 0, "input_symbol": 1, "target": 2, "output": [1]},
        {"source": 1, "input_symbol": 0, "target": 1, "output": [0]},
        {"source": 1, "input_symbol": 1, "target": 2, "output": [1]},
        {"source": 2, "input_symbol": 0, "target": 2, "output": [0]},
        {"source": 2, "input_symbol": 1, "target": 2, "output": [1]},
    ],
    "final_outputs": [
        {"state": 0, "output": []},
        {"state": 1, "output": []},
        {"state": 2, "output": [0]},
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
        operation_id="transducer.subsequential.trim.compute",
        title="Trim a subsequential transducer to live states",
        description="Restrict one subsequential transducer to states that are both "
        "reachable from the initial state and able to reach a final-output state. "
        "Return the restricted transducer with exact old-to-new and new-to-old "
        "state maps. The restriction preserves the partial function: every defined "
        "input word produces the same output word as before.",
        request_type=TrimRequest,
        result_type=TrimResult,
        run=compute_trim,
        tags=("transducer", "subsequential", "trim", "exact"),
        discovery_terms=("trim", "reachable states", "coaccessible states"),
        examples=(
            OperationExample(
                name="drop_dead_and_unreachable_states",
                description="Trim a three-state transducer with one reachable dead "
                "state and one unreachable state down to its live single-state "
                "restriction.",
                input={"transducer": _TRIM_SOURCE},
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.subsequential.minimize.compute",
        title="Minimize a subsequential transducer",
        description="Merge the coarsest exact-output bisimulation of one "
        "subsequential transducer by partition refinement, preserving the "
        "realized partial function exactly. Return the quotient transducer "
        "with old-to-new and new-to-old state maps, the merged partition, "
        "a Myhill-Nerode-style state-distinguishability table, and a replay "
        "of the shipped run semantics on every word up to a bounded sample "
        "length.",
        request_type=MinimizeRequest,
        result_type=MinimizeResult,
        run=compute_minimize,
        tags=("transducer", "subsequential", "minimization", "exact"),
        discovery_terms=("minimize", "minimization", "state merging", "bisimulation"),
        examples=(
            OperationExample(
                name="merge_two_equivalent_states",
                description="Minimize a three-state transducer whose states 0 "
                "and 1 realize the same partial function down to two states.",
                input={"transducer": _MINIMIZE_SOURCE, "sample_max_length": 3},
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
