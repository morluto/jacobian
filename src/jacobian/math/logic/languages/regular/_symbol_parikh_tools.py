"""Symbol-level Parikh profile declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.languages.regular._symbol_parikh import (
    SymbolParikhProfileRequest,
    SymbolParikhProfileResult,
    symbol_parikh_profile,
)

SYMBOL_PARIKH_PROFILE_OPERATION = MathTool(
    operation_id="regular_language.symbol_parikh_profile.compute",
    title="Compute an accepted-word symbol Parikh profile",
    description=(
        "Return the exact histogram of alphabet-symbol count vectors among accepted "
        "words of one length, retaining the DFA and ordered alphabet axis."
    ),
    request_type=SymbolParikhProfileRequest,
    result_type=SymbolParikhProfileResult,
    run=symbol_parikh_profile,
    tags=("regular-language", "dfa", "parikh", "symbols", "exact"),
    examples=(
        OperationExample(
            name="binary_words_ending_in_one",
            description="Profile binary words of length three that end in one.",
            input={
                "dfa": {
                    "state_count": 2,
                    "alphabet_size": 2,
                    "transitions": [
                        {"source": 0, "symbol": 0, "target": 0},
                        {"source": 0, "symbol": 1, "target": 1},
                        {"source": 1, "symbol": 0, "target": 0},
                        {"source": 1, "symbol": 1, "target": 1},
                    ],
                    "initial_state": 0,
                    "accepting_states": [1],
                },
                "word_length": 3,
            },
        ),
    ),
)
