"""Typed declarations for the word-cube combinatorial-line hypergraph."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.word_cube_hypergraph._models import (
    WordCubeRequest,
    WordCubeResult,
)
from jacobian.math.combinatorics.word_cube_hypergraph.operations import (
    compute_word_cube_hypergraph,
)


def _compute(request: WordCubeRequest) -> WordCubeResult:
    return compute_word_cube_hypergraph(request.alphabet_size, request.dimension)


TOOLS: MathTools = (
    MathTool(
        operation_id="words.combinatorial_line_hypergraph.compute",
        title="Construct finite word-cube combinatorial-line hypergraphs",
        description=(
            "Given alphabet cardinality q and dimension d, return the complete "
            "canonical q-uniform hypergraph of combinatorial lines in [q]^d."
        ),
        request_type=WordCubeRequest,
        result_type=WordCubeResult,
        run=_compute,
        tags=("words", "combinatorial", "hales_jewett", "exact"),
        examples=(
            example(
                "q2_d2",
                "Combinatorial lines in [2]^2.",
                {"alphabet_size": 2, "dimension": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
