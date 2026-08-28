"""Latin square operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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
    return LatinSquareCheckResult(is_latin=is_latin_square(request.square))


def compute_orthogonality(request: OrthogonalityRequest) -> OrthogonalityResult:
    is_orthogonal, pair_count = orthogonality_profile(
        request.square_a, request.square_b
    )
    return OrthogonalityResult(
        is_orthogonal=is_orthogonal,
        pair_count=pair_count,
    )


def compute_latin_square_transpose(
    request: TransposeRequest,
) -> LatinSquareTransposeResult:
    return LatinSquareTransposeResult(transposed=transpose(request.square))


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "latin_square.check",
        "Check if a matrix is a Latin square",
        "Verify that each row and column contains every symbol 0..n-1 exactly once.",
        LatinSquareRequest,
        LatinSquareCheckResult,
        compute_latin_square_check,
        "latin-square",
        "verification",
        "exact",
        examples=(
            example(
                "z2_latin_square",
                "Check the 2x2 Latin square [[0,1],[1,0]].",
                {
                    "square": {
                        "order": 2,
                        "cells": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
    _op(
        "latin_square.orthogonality.check",
        "Check orthogonality of two Latin squares",
        "Check whether two Latin squares of the same order are orthogonal, "
        "i.e., all ordered pairs of entries are distinct.",
        OrthogonalityRequest,
        OrthogonalityResult,
        compute_orthogonality,
        "latin-square",
        "orthogonality",
        "exact",
        examples=(
            example(
                "orthogonal_z2",
                "Check orthogonality of [[0,1],[1,0]] and [[0,1],[1,0]].",
                {
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
    _op(
        "latin_square.transpose.compute",
        "Transpose a Latin square",
        "Swap rows and columns of a Latin square.",
        TransposeRequest,
        LatinSquareTransposeResult,
        compute_latin_square_transpose,
        "latin-square",
        "transpose",
        "exact",
        examples=(
            example(
                "transpose_z2",
                "Transpose [[0,1],[1,0]].",
                {
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
