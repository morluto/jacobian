"""Oriented-matroid operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.oriented_matroids._models import (
    ChirotopeCheckRequest,
    ChirotopeCheckResult,
)
from jacobian.math.oriented_matroids._operations import check_chirotope

_ALTERNATING_RANK3_EXAMPLE: dict[str, Any] = {
    "chirotope": {
        "ground_size": 4,
        "entries": [
            {"triple": [0, 1, 2], "sign": 1},
            {"triple": [0, 1, 3], "sign": 1},
            {"triple": [0, 2, 3], "sign": 1},
            {"triple": [1, 2, 3], "sign": 1},
        ],
    }
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="oriented_matroid.chirotope.check",
        title="Check a complete uniform rank-3 chirotope",
        description=(
            "Exhaustively validate the complete rank-3 B2 exchange axiom of one "
            "canonical uniform rank-3 sign table. The source must be the "
            "lexicographic table on indexed elements 0..n-1 with n <= 10; the "
            "result gives exact checked counts and the first obstruction, if any. "
            "Nonalternating or zero presentations are rejected at request "
            "admission. Each public invocation reserves one producer scan, while "
            "the reported count describes that exact scan."
        ),
        request_type=ChirotopeCheckRequest,
        result_type=ChirotopeCheckResult,
        run=check_chirotope,
        tags=("oriented-matroid", "chirotope", "exact"),
        examples=(
            example(
                "alternating_rank3_four_elements",
                "Check the alternating rank-3 chirotope on four indexed elements; "
                "the input lists every increasing triple in lexicographic order.",
                _ALTERNATING_RANK3_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
