"""MathTool declaration for exact subsequential output lengths."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers.output_length._models import (
    SubsequentialOutputLengthRequest,
    SubsequentialOutputLengthResult,
)
from jacobian.math.logic.automata.transducers.output_length.operations import (
    subsequential_output_length,
)


def compute_output_length(
    request: SubsequentialOutputLengthRequest,
) -> SubsequentialOutputLengthResult:
    return subsequential_output_length(
        getattr(request, "transducer", None), getattr(request, "word", None)
    )


_MACHINE = {
    "input_alphabet_size": 1,
    "output_alphabet_size": 1,
    "state_count": 1,
    "initial_state": 0,
    "transitions": [{"source": 0, "input_symbol": 0, "target": 0, "output": [0, 0]}],
    "final_outputs": [{"state": 0, "output": [0]}],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="transducer.subsequential.output_length.compute",
        title="Compute a subsequential transducer output length",
        description=(
            "Return the exact length of the output on one bounded input word "
            "without constructing the output word. Preserve undefined-transition "
            "and nonfinal-domain outcomes, including the transition-output length "
            "emitted before failure."
        ),
        request_type=SubsequentialOutputLengthRequest,
        result_type=SubsequentialOutputLengthResult,
        run=compute_output_length,
        tags=("transducer", "subsequential", "output-length", "exact"),
        examples=(
            OperationExample(
                name="three_output_symbols",
                description=(
                    "A transition emits two symbols and the final output adds one, "
                    "so the total output length is three."
                ),
                input={"transducer": _MACHINE, "word": [0]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
