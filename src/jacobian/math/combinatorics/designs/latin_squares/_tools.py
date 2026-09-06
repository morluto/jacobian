"""Latin square operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.designs.latin_squares._models import (
    LatinSquareCheckResult,
    LatinSquareRequest,
    LatinSquareTransposeResult,
    OrthogonalityRequest,
    OrthogonalityResult,
    TransposeRequest,
)
from jacobian.math.combinatorics.designs.latin_squares.operations import (
    is_latin_square,
    orthogonality_profile,
    transpose,
)


def compute_latin_square_check(request: LatinSquareRequest) -> LatinSquareCheckResult:
    return LatinSquareCheckResult(
        square=request.square,
        is_latin=is_latin_square(request.square),
    )


def compute_orthogonality(request: OrthogonalityRequest) -> OrthogonalityResult:
    is_orthogonal, pair_count = orthogonality_profile(
        request.square_a, request.square_b
    )
    return OrthogonalityResult(
        square_a=request.square_a,
        square_b=request.square_b,
        is_orthogonal=is_orthogonal,
        pair_count=pair_count,
    )


def compute_latin_square_transpose(
    request: TransposeRequest,
) -> LatinSquareTransposeResult:
    return LatinSquareTransposeResult.model_construct(
        transposed=transpose(request.square),
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="latin_square.check",
        title="Check if a matrix is a Latin square",
        description="Verify that each row and column contains every symbol 0..n-1 exactly once.",
        request_type=LatinSquareRequest,
        result_type=LatinSquareCheckResult,
        run=compute_latin_square_check,
        tags=("latin-square", "verification", "exact"),
        examples=(
            OperationExample(
                name="z2_latin_square",
                description="Check the 2x2 Latin square [[0,1],[1,0]].",
                input={
                    "square": {
                        "order": 2,
                        "cells": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="latin_square.orthogonality.check",
        title="Check orthogonality of two Latin squares",
        description="Check whether two Latin squares of the same order are orthogonal, "
        "i.e., all ordered pairs of entries are distinct.",
        request_type=OrthogonalityRequest,
        result_type=OrthogonalityResult,
        run=compute_orthogonality,
        tags=("latin-square", "orthogonality", "exact"),
        examples=(
            OperationExample(
                name="orthogonal_z2",
                description="Check orthogonality of [[0,1],[1,0]] and [[0,1],[1,0]].",
                input={
                    "square_a": {
                        "order": 2,
                        "cells": [[0, 1], [1, 0]],
                    },
                    "square_b": {
                        "order": 2,
                        "cells": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="latin_square.transpose.compute",
        title="Transpose a Latin square",
        description="Swap rows and columns of a Latin square.",
        request_type=TransposeRequest,
        result_type=LatinSquareTransposeResult,
        run=compute_latin_square_transpose,
        tags=("latin-square", "transpose", "exact"),
        examples=(
            OperationExample(
                name="transpose_z2",
                description="Transpose [[0,1],[1,0]].",
                input={
                    "square": {
                        "order": 2,
                        "cells": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
