"""Catalog operation for regular output-tape restriction."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    RestrictRationalOutputRequest,
    RestrictRationalOutputResult,
)
from jacobian.math.logic.automata.transducers.output_restriction.operations import (
    restrict_rational_output,
)

_EXAMPLE_RELATION = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 2,
    "input_alphabet": {"symbols": ["i", "j"]},
    "output_alphabet": {"symbols": ["x", "y"]},
    "state_count": 2,
    "initial_states": [0],
    "accepting_states": [1],
    "edges": [
        {"source": 0, "target": 1, "input_label": [0], "output_label": [0]},
        {"source": 0, "target": 1, "input_label": [1], "output_label": [1]},
    ],
}
_EXAMPLE_DFA = {
    "state_count": 2,
    "alphabet_size": 2,
    "transitions": [
        {"source": 0, "symbol": 0, "target": 1},
        {"source": 0, "symbol": 1, "target": 0},
        {"source": 1, "symbol": 0, "target": 1},
        {"source": 1, "symbol": 1, "target": 0},
    ],
    "initial_state": 0,
    "accepting_states": [1],
}


def _run_restrict_output(
    request: RestrictRationalOutputRequest,
) -> RestrictRationalOutputResult:
    return restrict_rational_output(
        request.transducer,
        request.output_language,
        request.output_alphabet,
    )


TOOLS = (
    MathTool(
        operation_id="transducer.relation.restrict_output.compute",
        title="Restrict a rational transducer by an output language",
        description=(
            "Intersect a finite rational relation with a total DFA language on "
            "its output tape. The exact reachable product retains transducer "
            "edge alternatives, advances the DFA over complete output labels, "
            "and returns state and source-edge transports."
        ),
        request_type=RestrictRationalOutputRequest,
        result_type=RestrictRationalOutputResult,
        run=_run_restrict_output,
        tags=("transducer", "rational-relation", "output-restriction", "exact"),
        discovery_terms=(
            "restrict rational relation output",
            "transducer output language intersection",
            "filter transducer outputs by DFA",
            "regular output restriction",
        ),
        examples=(
            OperationExample(
                name="keep_outputs_ending_in_x",
                description=(
                    "Retain precisely relation paths whose output word is accepted "
                    "by the supplied DFA."
                ),
                input={
                    "transducer": _EXAMPLE_RELATION,
                    "output_language": _EXAMPLE_DFA,
                    "output_alphabet": {"symbols": ["x", "y"]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
