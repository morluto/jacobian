"""Word-cube combinatorial-line hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.word_cubes._models import (
    CombinatorialLineHypergraphRequest,
    CombinatorialLineHypergraphResult,
)
from jacobian.math.combinatorics.word_cubes.operations import (
    construct_combinatorial_line_hypergraph,
)


def compute_combinatorial_line_hypergraph(
    request: CombinatorialLineHypergraphRequest,
) -> CombinatorialLineHypergraphResult:
    return construct_combinatorial_line_hypergraph(
        request.alphabet_size, request.dimension
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="words.combinatorial_line_hypergraph.compute",
        title="Construct the combinatorial-line hypergraph of a word cube",
        description=(
            "Construct the complete canonical q-uniform hypergraph of "
            "combinatorial lines in the word cube [q]^d. Vertices are all "
            "length-d words over the alphabet {0,...,q-1}, and edges "
            "correspond to patterns with at least one wildcard coordinate."
        ),
        request_type=CombinatorialLineHypergraphRequest,
        result_type=CombinatorialLineHypergraphResult,
        run=compute_combinatorial_line_hypergraph,
        tags=("combinatorics", "words", "exact"),
        examples=(
            OperationExample(
                name="binary_cube_dim2",
                description="The word cube [2]^2 has 4 vertices and 5 combinatorial lines.",
                input={"alphabet_size": 2, "dimension": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
