"""Catalog declaration for subsequential coaccessibility witnesses."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers.coaccessible_states._models import (
    CoaccessibleStatesRequest,
    CoaccessibleStateWitnesses,
)
from jacobian.math.logic.automata.transducers.coaccessible_states.operations import (
    coaccessible_state_witnesses,
)


def compute_coaccessible_states(
    request: CoaccessibleStatesRequest,
) -> CoaccessibleStateWitnesses:
    return coaccessible_state_witnesses(request.transducer)


_TRANSDUCER = {
    "input_alphabet_size": 2,
    "output_alphabet_size": 3,
    "state_count": 3,
    "initial_state": 0,
    "transitions": [
        {"source": 0, "input_symbol": 1, "target": 2, "output": [1]},
        {"source": 0, "input_symbol": 0, "target": 1, "output": []},
        {"source": 2, "input_symbol": 0, "target": 1, "output": [2]},
    ],
    "final_outputs": [{"state": 1, "output": [0]}],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="transducer.subsequential.coaccessible_states.compute",
        title="Find successful continuations from transducer states",
        description=(
            "For each state from which a final-output state is reachable, return "
            "a shortest input suffix, breaking ties by input-symbol order, and "
            "the exact transition and final output produced by that suffix."
        ),
        request_type=CoaccessibleStatesRequest,
        result_type=CoaccessibleStateWitnesses,
        run=compute_coaccessible_states,
        tags=("transducer", "subsequential", "coaccessibility", "exact"),
        discovery_terms=("coaccessible states", "successful suffix", "final output"),
        examples=(
            OperationExample(
                name="shortest_successful_suffixes",
                description="Return a shortest successful continuation per live state.",
                input={"transducer": _TRANSDUCER},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
