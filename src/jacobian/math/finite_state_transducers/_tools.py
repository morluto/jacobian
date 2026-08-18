"""Finite-state transducer operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
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
from jacobian.math.finite_state_transducers._operations import (
    compute_compose,
    compute_identity,
    compute_relation_inverse,
    compute_relation_path_replay,
    compute_run,
    compute_trim,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


_TRANS1 = {
    "source": 0,
    "input_symbol": 0,
    "target": 0,
    "output": [1],
}
_TRANS2 = {
    "source": 0,
    "input_symbol": 1,
    "target": 1,
    "output": [0],
}
_TRANS3 = {
    "source": 1,
    "input_symbol": 0,
    "target": 0,
    "output": [0],
}
_TRANS4 = {
    "source": 1,
    "input_symbol": 1,
    "target": 1,
    "output": [1],
}

_RUN_EXAMPLE = {
    "transducer": {
        "input_alphabet_size": 2,
        "output_alphabet_size": 2,
        "state_count": 2,
        "initial_state": 0,
        "transitions": [_TRANS1, _TRANS2, _TRANS3, _TRANS4],
        "final_outputs": [{"state": 0, "output": []}, {"state": 1, "output": []}],
    },
    "word": [0, 1, 0],
}

_IDENTITY_EXAMPLE = {"alphabet_size": 2}

_TRIM_EXAMPLE = {"transducer": _RUN_EXAMPLE["transducer"]}

_INVERT_EXAMPLE = {
    "transducer": {
        "input_alphabet_size": 2,
        "output_alphabet_size": 2,
        "state_count": 1,
        "initial_states": [0],
        "accepting_states": [0],
        "edges": [
            {"source": 0, "target": 0, "input_label": [0], "output_label": [1]},
            {"source": 0, "target": 0, "input_label": [1], "output_label": [0]},
        ],
    },
}

_REPLAY_EXAMPLE = _INVERT_EXAMPLE.copy()


FINITE_STATE_TRANSDUCER_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "transducer.subsequential.run.compute",
        "Run a subsequential transducer on a word",
        "Execute a deterministic subsequential transducer and return the exact "
        "output, distinguishing undefined transitions, nonfinal domain states, "
        "and successful output.",
        SubseqRunRequest,
        SubseqRunResult,
        compute_run,
        "transducer",
        "subsequential",
        "exact",
        examples=(
            example(
                "binary_id_run",
                "Run a 2-symbol identity-like transducer on word [0,1,0].",
                _RUN_EXAMPLE,
            ),
        ),
    ),
    _op(
        "transducer.subsequential.identity.compute",
        "Construct the identity subsequential transducer",
        "Return the one-state subsequential transducer that outputs each "
        "consumed input symbol unchanged with an empty final output.",
        IdentityRequest,
        IdentityResult,
        compute_identity,
        "transducer",
        "subsequential",
        "identity",
        "exact",
        examples=(
            example(
                "binary_identity",
                "Identity transducer on a 2-symbol alphabet.",
                _IDENTITY_EXAMPLE,
            ),
        ),
    ),
    _op(
        "transducer.subsequential.compose.compute",
        "Compose two subsequential transducers",
        "Return the subsequential transducer computing U∘T where T:A*->B* and "
        "U:B*->C*. The composition simulates U over each T transition output.",
        ComposeRequest,
        ComposeResult,
        compute_compose,
        "transducer",
        "subsequential",
        "composition",
        "exact",
        examples=(
            example(
                "compose_identity_flip",
                "Compose identity with bit-flip transducer.",
                {
                    "first": {
                        "input_alphabet_size": 2,
                        "output_alphabet_size": 2,
                        "state_count": 1,
                        "initial_state": 0,
                        "transitions": [
                            {"source": 0, "input_symbol": 0, "target": 0, "output": [0]},
                            {"source": 0, "input_symbol": 1, "target": 0, "output": [1]},
                        ],
                        "final_outputs": [{"state": 0, "output": []}],
                    },
                    "second": {
                        "input_alphabet_size": 2,
                        "output_alphabet_size": 2,
                        "state_count": 1,
                        "initial_state": 0,
                        "transitions": [
                            {"source": 0, "input_symbol": 0, "target": 0, "output": [1]},
                            {"source": 0, "input_symbol": 1, "target": 0, "output": [0]},
                        ],
                        "final_outputs": [{"state": 0, "output": []}],
                    },
                },
            ),
        ),
    ),
    _op(
        "transducer.subsequential.trim.compute",
        "Trim a subsequential transducer to reachable and coaccessible states",
        "Restrict a subsequential transducer to states that are both reachable "
        "from the initial state and can reach a final-output state.",
        TrimRequest,
        TrimResult,
        compute_trim,
        "transducer",
        "subsequential",
        "trim",
        "exact",
        examples=(
            example(
                "trim_dead_states",
                "Trim a transducer with dead states.",
                _TRIM_EXAMPLE,
            ),
        ),
    ),
    _op(
        "transducer.relation.inverse.compute",
        "Invert a rational transducer",
        "Swap input and output alphabets and every edge label pair of a "
        "rational transducer, producing the inverse relation R^{-1}.",
        RelationInverseRequest,
        RelationInverseResult,
        compute_relation_inverse,
        "transducer",
        "rational",
        "inverse",
        "exact",
        examples=(
            example(
                "bit_flip_inverse",
                "Invert a bit-flip rational transducer.",
                _INVERT_EXAMPLE,
            ),
        ),
    ),
    _op(
        "transducer.relation.path.replay.compute",
        "Replay a candidate edge path of a rational transducer",
        "Replay a given edge path through a rational transducer and return the "
        "concatenated input/output words, state trace, and acceptance status.",
        RelationPathReplayRequest,
        RelationPathReplayResult,
        compute_relation_path_replay,
        "transducer",
        "rational",
        "path-replay",
        "exact",
        examples=(
            example(
                "replay_simple_path",
                "Replay a simple edge path.",
                {"transducer": _REPLAY_EXAMPLE["transducer"], "edge_path": [0, 0]},
            ),
        ),
    ),
)

TOOLS = FINITE_STATE_TRANSDUCER_OPERATIONS

__all__ = ["TOOLS"]
