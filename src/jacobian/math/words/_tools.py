"""Public combinatorics-on-words operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.words._models import (
    FactorsLengthRequest,
    FactorsLengthResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    PeriodsRequest,
    PeriodsResult,
)
from jacobian.math.words._operations import (
    compute_factors_length,
    compute_incidence_matrix,
    compute_periods,
)

WORDS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="word.factors.length.compute",
        version="1",
        title="Compute all factors of one length",
        description=(
            "Enumerate every distinct contiguous factor of the requested length, "
            "in first-occurrence order, with all zero-based occurrence positions."
        ),
        request_type=FactorsLengthRequest,
        result_type=FactorsLengthResult,
        run=compute_factors_length,
        tags=("combinatorics", "words", "factors", "exact", "complete"),
        examples=(
            example(
                "abaab_factors_2",
                "Enumerate all length-two factors of abaab.",
                {
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "a", "b"],
                    },
                    "factor_length": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word.periods.compute",
        version="1",
        title="Compute all periods of a word",
        description=(
            "Return every positive overlap period and decide whether the word is "
            "a nontrivial integer power. An empty word has no positive periods and "
            "is not primitive."
        ),
        request_type=PeriodsRequest,
        result_type=PeriodsResult,
        run=compute_periods,
        tags=("combinatorics", "words", "periods", "exact", "complete"),
        examples=(
            example(
                "ababab_periods",
                "Compute the complete period profile of ababab.",
                {
                    "word": {
                        "alphabet": ["a", "b"],
                        "letters": ["a", "b", "a", "b", "a", "b"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="word_morphism.incidence_matrix.compute",
        version="1",
        title="Compute a word-morphism incidence matrix",
        description=(
            "Compute the exact matrix whose target-symbol rows and source-symbol "
            "columns count symbols in each morphism image."
        ),
        request_type=IncidenceMatrixRequest,
        result_type=IncidenceMatrixResult,
        run=compute_incidence_matrix,
        tags=("combinatorics", "words", "morphism", "matrix", "exact"),
        examples=(
            example(
                "fibonacci_matrix",
                "Compute the incidence matrix of a->ab and b->a.",
                {
                    "morphism": {
                        "source_alphabet": ["a", "b"],
                        "target_alphabet": ["a", "b"],
                        "images": [["a", "b"], ["a"]],
                    }
                },
            ),
        ),
    ),
)

TOOLS = WORDS_OPERATIONS

__all__ = ["TOOLS"]
