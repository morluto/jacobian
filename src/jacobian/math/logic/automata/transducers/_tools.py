"""Finite-state transducer operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers._models import (
    ComposeRequest,
    ComposeResult,
    MinimizeRequest,
    MinimizeResult,
    RationalRelationInverseRequest,
    RationalRelationProjectionRequest,
    ReachableStatesRequest,
    ReachableStatesResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqIdentityRequest,
    SubseqRunRequest,
    SubseqRunResult,
    TrimRequest,
    TrimResult,
    WordMorphismToSubseqRequest,
)
from jacobian.math.logic.automata.transducers.operations import (
    compose_subsequential,
    identity_transducer,
    invert_rational,
    minimize_subsequential,
    project_rational_relation,
    reachable_state_witnesses,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
    word_morphism_to_subsequential,
)
from jacobian.math.logic.automata.transducers.values import (
    RationalTransducer,
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular.values import NFA


def compute_run(request: SubseqRunRequest) -> SubseqRunResult:
    return run_subsequential(request.transducer, request.word)


def compute_identity(request: SubseqIdentityRequest) -> SubsequentialTransducer:
    return identity_transducer(
        len(request.alphabet.symbols),
        alphabet=request.alphabet,
        alphabet_id=request.alphabet_id,
    )


def compute_word_morphism_to_subsequential(
    request: WordMorphismToSubseqRequest,
) -> SubsequentialTransducer:
    return word_morphism_to_subsequential(request.morphism)


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


def compute_reachable_states(
    request: ReachableStatesRequest,
) -> ReachableStatesResult:
    return reachable_state_witnesses(request.transducer)


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


def compute_relation_inverse(
    request: RationalRelationInverseRequest,
) -> RationalTransducer:
    return invert_rational(request.transducer)


def compute_relation_projection(
    request: RationalRelationProjectionRequest,
) -> NFA:
    return project_rational_relation(request.transducer, request.tape)


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
        operation_id="transducer.subsequential.identity.compute",
        title="Construct the identity subsequential transducer",
        description=(
            "Construct the total one-state identity function on an explicit "
            "ordered finite alphabet. Each input symbol is emitted unchanged; "
            "the final output is empty. Input and output retain the same alphabet "
            "context and optional identity."
        ),
        request_type=SubseqIdentityRequest,
        result_type=SubsequentialTransducer,
        run=compute_identity,
        tags=("transducer", "subsequential", "identity", "exact"),
        examples=(
            OperationExample(
                name="binary_identity",
                description="Construct identity on ordered alphabet (a,b).",
                input={"alphabet": {"symbols": ["a", "b"]}, "alphabet_id": "binary"},
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.subsequential.from_word_morphism.compute",
        title="Represent a word morphism as a subsequential transducer",
        description=(
            "Construct the one-state total subsequential transducer for a finite "
            "word morphism. Source and target symbol orders are preserved, each "
            "source symbol has one transition carrying its exact image, and the "
            "final output is empty. The transducer carrier admits at most 32 "
            "symbols per alphabet and 512 symbols per image."
        ),
        request_type=WordMorphismToSubseqRequest,
        result_type=SubsequentialTransducer,
        run=compute_word_morphism_to_subsequential,
        tags=("transducer", "subsequential", "word-morphism", "exact"),
        examples=(
            OperationExample(
                name="morphism_with_empty_image",
                description=(
                    "Represent a->xy and b->empty as total transitions; b remains "
                    "defined and emits the empty word."
                ),
                input={
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["x", "y"],
                        "images": [["x", "y"], []],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.subsequential.run.compute",
        title="Run a subsequential transducer on a word",
        description=(
            "Execute one exact bounded run and return its state-after-prefix trace, "
            "transition outputs, cumulative outputs, separate final output, and "
            "complete output. Successful empty output, undefined transitions, and "
            "termination in a nonfinal state remain distinct."
        ),
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
        description="Normalize one subsequential transducer by pushing each live "
        "state's common output prefix onto its incoming transitions, then merge "
        "the coarsest exact-output bisimulation of the normalized machine by "
        "partition refinement, preserving the realized partial function exactly. "
        "Because pushing can only coarsen the bisimulation, the quotient is the "
        "state-minimal transducer for this value model, which has no separate "
        "initial output. Return the quotient transducer with old-to-new and "
        "new-to-old state maps, the merged partition, a Myhill-Nerode-style "
        "state-distinguishability table, and a replay of the shipped run "
        "semantics on every word up to a bounded sample length.",
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
        operation_id="transducer.subsequential.reachable_states.compute",
        title="Find reachable states and shortest output paths",
        description=(
            "Return one shortest input word to each reachable state, its exact "
            "state trace, and the concatenated transition output. Equal-length "
            "ties use lexicographically least input words. Final outputs are "
            "excluded because a witness reaches a state at an input prefix."
        ),
        request_type=ReachableStatesRequest,
        result_type=ReachableStatesResult,
        run=compute_reachable_states,
        tags=("transducer", "subsequential", "reachability", "exact"),
        discovery_terms=("reachable states", "state reachability", "shortest path"),
        examples=(
            OperationExample(
                name="reachable_shortest_witnesses",
                description=(
                    "Return the shortest input and emitted transition word "
                    "reaching each state of a three-state transducer."
                ),
                input={"transducer": _MINIMIZE_SOURCE},
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.relation.inverse.compute",
        title="Invert a finite rational relation",
        description=(
            "Return the same finite-state relation with input and output "
            "labels and their alphabet parents swapped. States, initial and "
            "accepting sets, and edge order are preserved. Applying the "
            "operation twice returns the original canonical relation."
        ),
        request_type=RationalRelationInverseRequest,
        result_type=RationalTransducer,
        run=compute_relation_inverse,
        tags=("transducer", "rational-relation", "inverse", "exact"),
        examples=(
            OperationExample(
                name="invert_two_pair_relation",
                description="Swap each input/output word pair in a two-edge relation.",
                input={
                    "transducer": {
                        "input_alphabet_size": 2,
                        "output_alphabet_size": 2,
                        "state_count": 1,
                        "initial_states": [0],
                        "accepting_states": [0],
                        "edges": [
                            {
                                "source": 0,
                                "target": 0,
                                "input_label": [0],
                                "output_label": [1],
                            },
                            {
                                "source": 0,
                                "target": 0,
                                "input_label": [1],
                                "output_label": [0],
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="transducer.relation.projection.compute",
        title="Project a finite rational relation to one tape",
        description=(
            "Return an epsilon-NFA accepting exactly the selected tape words "
            "that occur on accepting paths. The relation may be nondeterministic: "
            "different accepting paths and output choices remain possible. "
            "Multi-symbol labels expand to paths and empty labels to epsilon "
            "edges. Alphabet symbols, explicit parent, and optional identity "
            "are preserved; result expansion is admitted before construction."
        ),
        request_type=RationalRelationProjectionRequest,
        result_type=NFA,
        run=compute_relation_projection,
        tags=("transducer", "rational-relation", "projection", "exact"),
        examples=(
            OperationExample(
                name="project_relation_output",
                description=(
                    "Project accepting pairs (a,xy) and (ba,x) to their output "
                    "language {xy,x}."
                ),
                input={
                    "tape": "output",
                    "transducer": {
                        "input_alphabet_size": 2,
                        "output_alphabet_size": 2,
                        "state_count": 3,
                        "initial_states": [0],
                        "accepting_states": [2],
                        "edges": [
                            {
                                "source": 0,
                                "target": 2,
                                "input_label": [0],
                                "output_label": [0, 1],
                            },
                            {
                                "source": 0,
                                "target": 1,
                                "input_label": [1],
                                "output_label": [0],
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "input_label": [0],
                                "output_label": [],
                            },
                        ],
                    },
                },
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
